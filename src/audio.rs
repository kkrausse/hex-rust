use std::ops::Add;
#[cfg(target_os = "macos")]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TryRecvError};
use std::time::{Duration, Instant};

use color_eyre::eyre::{Result, WrapErr, eyre};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, SizedSample, Stream, StreamConfig};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CaptureInstant(u64);

impl CaptureInstant {
    pub const ZERO: Self = Self(0);

    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    /// Milliseconds on the capture clock; release-latency logs share it.
    pub fn as_millis(self) -> u64 {
        self.0 / 1_000_000
    }

    #[cfg(target_os = "macos")]
    pub fn now() -> Self {
        unsafe extern "C" {
            fn mach_absolute_time() -> u64;
        }
        Self::from_mach_ticks(unsafe { mach_absolute_time() })
    }

    #[cfg(target_os = "macos")]
    pub fn from_mach_ticks(ticks: u64) -> Self {
        let Some((numerator, denominator)) = mach_timebase() else {
            return Self::ZERO;
        };
        Self(scale_mach_ticks(ticks, numerator, denominator))
    }

    pub fn from_stream(instant: cpal::StreamInstant) -> Self {
        Self(u64::try_from(instant.as_nanos()).unwrap_or(u64::MAX))
    }

    pub fn checked_add(self, duration: Duration) -> Option<Self> {
        let nanos = u64::try_from(duration.as_nanos()).ok()?;
        self.0.checked_add(nanos).map(Self)
    }

    pub fn checked_sub(self, duration: Duration) -> Option<Self> {
        let nanos = u64::try_from(duration.as_nanos()).ok()?;
        self.0.checked_sub(nanos).map(Self)
    }

    pub fn checked_duration_since(self, earlier: Self) -> Option<Duration> {
        self.0.checked_sub(earlier.0).map(Duration::from_nanos)
    }

    pub fn duration_since(self, earlier: Self) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(earlier.0))
    }

    pub fn sub_frames(self, frames: usize, sample_rate: u32) -> Self {
        self.checked_sub(duration_for_frames(frames, sample_rate))
            .unwrap_or(self)
    }
}

pub fn duration_for_frames(frames: usize, sample_rate: u32) -> Duration {
    let nanos = frames as u128 * 1_000_000_000 / u128::from(sample_rate);
    Duration::from_nanos(nanos as u64)
}

#[cfg(target_os = "macos")]
fn scale_mach_ticks(ticks: u64, numerator: u32, denominator: u32) -> u64 {
    let nanos = u128::from(ticks) * u128::from(numerator) / u128::from(denominator);
    u64::try_from(nanos).unwrap_or(u64::MAX)
}

#[cfg(target_os = "macos")]
fn mach_timebase() -> Option<(u32, u32)> {
    #[repr(C)]
    struct MachTimebaseInfo {
        numer: u32,
        denom: u32,
    }

    unsafe extern "C" {
        fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
    }

    static TIMEBASE: OnceLock<Option<(u32, u32)>> = OnceLock::new();
    *TIMEBASE.get_or_init(|| {
        let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
        let status = unsafe { mach_timebase_info(&mut info) };
        (status == 0 && info.denom != 0).then_some((info.numer, info.denom))
    })
}

impl Add<Duration> for CaptureInstant {
    type Output = Self;

    fn add(self, duration: Duration) -> Self::Output {
        self.checked_add(duration)
            .expect("capture instant overflowed")
    }
}

pub struct AudioInput {
    // Tests can deliver PCM through channels without opening a native microphone.
    _stream: Option<Stream>,
    chunks: Receiver<(Vec<f32>, CaptureInstant)>,
    pub sample_rate: u32,
    pub device_name: String,
    stream_errors: Receiver<String>,
}

pub enum AudioInputEvent {
    Chunk {
        samples: Vec<f32>,
        captured_through: CaptureInstant,
    },
    Timeout,
    StreamFailed(String),
}

type InputOpenResult = (u64, Result<AudioInput, String>);

pub struct RecoveringAudioInput {
    input: Option<AudioInput>,
    device_override: Option<String>,
    microphone_priority: Vec<String>,
    active_revision: u64,
    recovery: MicrophoneRecovery,
    replacement: Option<Receiver<InputOpenResult>>,
    topology: InputTopology,
}

pub enum RecoveringAudioInputEvent {
    Chunk {
        samples: Vec<f32>,
        captured_through: CaptureInstant,
    },
    Timeout,
    Interrupted,
    Reopened,
    OpenFailed(String),
}

const MICROPHONE_RETRY_INITIAL: Duration = Duration::from_millis(250);
const MICROPHONE_RETRY_MAX: Duration = Duration::from_secs(5);
/// CoreAudio reports one plug or default change as a burst of notifications.
const INPUT_TOPOLOGY_SETTLE: Duration = Duration::from_millis(300);
/// Without CoreAudio notifications, re-resolve the priority list this often.
const INPUT_TOPOLOGY_POLL: Duration = Duration::from_secs(2);

/// Bumped by CoreAudio whenever the device list or the default input changes.
static INPUT_TOPOLOGY_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Notices device-list and default-input changes so a warm stream can follow
/// the microphone priority list instead of keeping whatever it first opened.
struct InputTopology {
    seen: u64,
    changed_at: Option<Instant>,
    polled_at: Instant,
}

impl InputTopology {
    fn new(now: Instant) -> Self {
        watch_input_topology();
        Self {
            seen: INPUT_TOPOLOGY_GENERATION.load(Ordering::Acquire),
            changed_at: None,
            polled_at: now,
        }
    }

    fn take_change(&mut self, now: Instant) -> bool {
        if !watch_input_topology() {
            if now.duration_since(self.polled_at) < INPUT_TOPOLOGY_POLL {
                return false;
            }
            self.polled_at = now;
            return true;
        }
        let generation = INPUT_TOPOLOGY_GENERATION.load(Ordering::Acquire);
        if generation != self.seen {
            self.seen = generation;
            self.changed_at = Some(now);
        }
        if self
            .changed_at
            .is_some_and(|changed_at| now.duration_since(changed_at) >= INPUT_TOPOLOGY_SETTLE)
        {
            self.changed_at = None;
            return true;
        }
        false
    }
}

/// Registers process-lifetime CoreAudio listeners once. Returns whether they
/// are installed; callers poll instead when they are not.
#[cfg(target_os = "macos")]
fn watch_input_topology() -> bool {
    use std::ffi::c_void;
    use std::ptr::NonNull;

    use objc2_core_audio::{
        AudioObjectAddPropertyListener, AudioObjectID, AudioObjectPropertyAddress,
        kAudioHardwarePropertyDefaultInputDevice, kAudioHardwarePropertyDevices,
        kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal, kAudioObjectSystemObject,
    };

    unsafe extern "C-unwind" fn changed(
        _: AudioObjectID,
        _: u32,
        _: NonNull<AudioObjectPropertyAddress>,
        _: *mut c_void,
    ) -> i32 {
        INPUT_TOPOLOGY_GENERATION.fetch_add(1, Ordering::AcqRel);
        0
    }

    static WATCHING: OnceLock<bool> = OnceLock::new();
    *WATCHING.get_or_init(|| {
        [
            kAudioHardwarePropertyDevices,
            kAudioHardwarePropertyDefaultInputDevice,
        ]
        .into_iter()
        .all(|selector| {
            let address = AudioObjectPropertyAddress {
                mSelector: selector,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain,
            };
            let status = unsafe {
                AudioObjectAddPropertyListener(
                    kAudioObjectSystemObject as AudioObjectID,
                    NonNull::from(&address),
                    Some(changed),
                    std::ptr::null_mut(),
                )
            };
            if status != 0 {
                tracing::warn!(
                    status,
                    "could not watch microphone changes; polling instead"
                );
            }
            status == 0
        })
    })
}

#[cfg(target_os = "linux")]
fn watch_input_topology() -> bool {
    false
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum MicrophoneRecoveryReason {
    SelectionChanged,
    StreamFailed,
}

struct MicrophoneRecovery {
    next_attempt: Option<Instant>,
    delay: Duration,
    reason: Option<MicrophoneRecoveryReason>,
    target_revision: Option<u64>,
}

impl Default for MicrophoneRecovery {
    fn default() -> Self {
        Self {
            next_attempt: None,
            delay: MICROPHONE_RETRY_INITIAL,
            reason: None,
            target_revision: None,
        }
    }
}

impl MicrophoneRecovery {
    fn request_selection_change(&mut self, revision: u64, now: Instant) {
        if self.target_revision != Some(revision) {
            self.next_attempt = Some(now);
            self.delay = MICROPHONE_RETRY_INITIAL;
            self.target_revision = Some(revision);
            if self.reason != Some(MicrophoneRecoveryReason::StreamFailed) {
                self.reason = Some(MicrophoneRecoveryReason::SelectionChanged);
            }
        }
    }

    fn request_stream_recovery(&mut self, now: Instant) {
        if self.reason != Some(MicrophoneRecoveryReason::StreamFailed) {
            self.next_attempt = Some(now);
            self.delay = MICROPHONE_RETRY_INITIAL;
            self.reason = Some(MicrophoneRecoveryReason::StreamFailed);
        }
    }

    fn blocks_audio(&self) -> bool {
        self.reason == Some(MicrophoneRecoveryReason::StreamFailed)
    }

    fn should_attempt(&self, now: Instant) -> bool {
        self.next_attempt.is_some_and(|next| now >= next)
    }

    fn failed(&mut self, now: Instant) {
        self.next_attempt = Some(now + self.delay);
        self.delay = self.delay.saturating_mul(2).min(MICROPHONE_RETRY_MAX);
    }

    fn recovered(&mut self) {
        self.next_attempt = None;
        self.delay = MICROPHONE_RETRY_INITIAL;
        self.reason = None;
        self.target_revision = None;
    }
}

impl AudioInput {
    #[cfg(test)]
    pub fn channel_for_test() -> (Self, Sender<(Vec<f32>, CaptureInstant)>) {
        let (sender, chunks) = mpsc::channel();
        let (_, stream_errors) = mpsc::channel();
        (
            Self {
                _stream: None,
                chunks,
                stream_errors,
                sample_rate: 48_000,
                device_name: "Test microphone".into(),
            },
            sender,
        )
    }

    /// Opens the first available priority entry, else the system default,
    /// trying later choices when an earlier one fails to open.
    pub fn open(microphone_priority: &[String]) -> Result<Self> {
        Self::open_prioritized(microphone_priority, true)
    }

    fn open_prioritized(microphone_priority: &[String], fall_through: bool) -> Result<Self> {
        let host = cpal::default_host();
        let mut last_error = None;
        for (device, name) in input_candidates(&host, microphone_priority)? {
            match Self::open_device(device, name.clone()) {
                Ok(input) => return Ok(input),
                Err(error) if fall_through => {
                    tracing::warn!(%error, device = %name, "could not open microphone; trying the next choice");
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }
        }
        Err(last_error.unwrap_or_else(|| eyre!("no input device is available")))
    }

    pub fn open_named(name: &str) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .input_devices()
            .wrap_err("could not enumerate input devices")?
            .find(|device| device_name(device).as_deref() == Some(name))
            .ok_or_else(|| eyre!("microphone is unavailable: {name}"))?;
        Self::open_device(device, name.to_owned())
    }

    #[cfg(target_os = "linux")]
    pub fn open_matching(query: &str) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .input_devices()
            .wrap_err("could not enumerate input devices")?
            .find(|device| device_matches(device, query))
            .ok_or_else(|| eyre!("microphone is unavailable: {query}"))?;
        let name = device_name(&device).unwrap_or_else(|| query.to_owned());
        Self::open_device(device, name)
    }

    fn open_device(device: Device, device_name: String) -> Result<Self> {
        let supported = device
            .default_input_config()
            .wrap_err("could not read the input device configuration")?;
        let sample_format = supported.sample_format();
        let config: StreamConfig = supported.into();
        let sample_rate = config.sample_rate;
        let channels = usize::from(config.channels);
        let (sender, chunks) = mpsc::channel();
        let (error_sender, stream_errors) = mpsc::sync_channel(1);

        let stream = match sample_format {
            SampleFormat::F32 => build_stream(
                &device,
                &config,
                channels,
                sender,
                error_sender,
                |sample: f32| sample,
            )?,
            SampleFormat::I16 => build_stream(
                &device,
                &config,
                channels,
                sender,
                error_sender,
                |sample: i16| sample as f32 / i16::MAX as f32,
            )?,
            SampleFormat::U16 => build_stream(
                &device,
                &config,
                channels,
                sender,
                error_sender,
                |sample: u16| sample as f32 / 32768.0 - 1.0,
            )?,
            format => return Err(eyre!("unsupported microphone sample format: {format:?}")),
        };
        stream
            .play()
            .wrap_err("could not start microphone capture")?;

        Ok(Self {
            _stream: Some(stream),
            chunks,
            sample_rate,
            device_name,
            stream_errors,
        })
    }

    pub fn recv_timeout(&self, timeout: Duration) -> AudioInputEvent {
        // A stream failure reported before or during the wait outranks any
        // chunk delivered alongside it.
        if let Ok(error) = self.stream_errors.try_recv() {
            return AudioInputEvent::StreamFailed(error);
        }
        let received = self.chunks.recv_timeout(timeout);
        if let Ok(error) = self.stream_errors.try_recv() {
            return AudioInputEvent::StreamFailed(error);
        }
        match received {
            Ok((samples, captured_through)) => AudioInputEvent::Chunk {
                samples,
                captured_through,
            },
            Err(RecvTimeoutError::Timeout) => AudioInputEvent::Timeout,
            Err(RecvTimeoutError::Disconnected) => {
                AudioInputEvent::StreamFailed("microphone audio channel disconnected".into())
            }
        }
    }
}

impl RecoveringAudioInput {
    pub fn open(
        device_override: Option<&str>,
        selection_revision: u64,
        microphone_priority: &[String],
    ) -> Self {
        let mut input = Self::closed(device_override, selection_revision, microphone_priority);
        let opened = open_configured_input(device_override, microphone_priority, true);
        input.finish_initial_open(opened, Instant::now());
        input
    }

    fn finish_initial_open(&mut self, opened: Result<AudioInput>, now: Instant) {
        match opened {
            Ok(opened) => self.input = Some(opened),
            Err(error) => {
                tracing::warn!(%error, "could not open dictation microphone at startup; retrying");
                self.recovery.request_stream_recovery(now);
                self.recovery.failed(now);
            }
        }
    }

    pub fn closed(
        device_override: Option<&str>,
        selection_revision: u64,
        microphone_priority: &[String],
    ) -> Self {
        Self {
            input: None,
            device_override: device_override.map(str::to_owned),
            microphone_priority: microphone_priority.to_vec(),
            active_revision: selection_revision,
            recovery: MicrophoneRecovery::default(),
            replacement: None,
            topology: InputTopology::new(Instant::now()),
        }
    }

    #[cfg(test)]
    pub fn pending_for_test() -> (Self, Sender<InputOpenResult>) {
        let mut input = Self::closed(
            Some("HEX nonexistent microphone for pending open test"),
            0,
            &[],
        );
        let (sender, receiver) = mpsc::channel();
        input.replacement = Some(receiver);
        (input, sender)
    }

    pub fn request_open(&mut self) {
        if self.input.is_none() && self.replacement.is_none() && !self.is_recovering() {
            self.start_replacement();
        }
    }

    pub fn request_recovery(&mut self) {
        self.recovery.request_stream_recovery(Instant::now());
    }

    pub fn is_recovering(&self) -> bool {
        self.recovery.blocks_audio()
    }

    pub fn close(&mut self) {
        self.input = None;
        self.replacement = None;
        self.recovery.recovered();
    }

    pub fn is_open(&self) -> bool {
        self.input.is_some()
    }

    pub fn is_opening(&self) -> bool {
        self.replacement.is_some()
    }

    pub fn is_active(&self) -> bool {
        self.is_open() || self.is_opening() || self.is_recovering()
    }

    pub fn cancel_open(&mut self) {
        // Cancelling a capture must not cancel the warm microphone's recovery.
        if self.is_recovering() {
            return;
        }
        self.replacement = None;
        if self.input.is_none() {
            self.recovery.recovered();
        }
    }

    pub fn request_selection(&mut self, revision: u64, microphone_priority: &[String]) {
        if self.device_override.is_some() || revision == self.active_revision {
            return;
        }
        self.microphone_priority = microphone_priority.to_vec();
        if self.input.is_none() && !self.is_recovering() {
            let was_opening = self.replacement.take().is_some();
            self.active_revision = revision;
            self.recovery.recovered();
            if was_opening {
                self.start_replacement();
            }
            return;
        }
        self.recovery
            .request_selection_change(revision, Instant::now());
    }

    pub fn recv_timeout(
        &mut self,
        timeout: Duration,
        capture_idle: bool,
    ) -> RecoveringAudioInputEvent {
        let now = Instant::now();
        if let Some((opened_revision, result)) = self.poll_replacement() {
            let target_revision = self
                .recovery
                .target_revision
                .unwrap_or(self.active_revision);
            if opened_revision != target_revision {
                self.recovery.next_attempt = Some(now);
                return RecoveringAudioInputEvent::Timeout;
            }
            match result {
                Ok(replacement) if capture_idle || self.input.is_none() => {
                    self.input = Some(replacement);
                    self.active_revision = opened_revision;
                    self.recovery.recovered();
                    tracing::info!(device = %self.device_name(), "dictation microphone reopened");
                    return RecoveringAudioInputEvent::Reopened;
                }
                Ok(_) => {
                    self.recovery.failed(now);
                }
                Err(error) => {
                    if self.input.is_none() && !self.is_recovering() {
                        self.recovery.recovered();
                        return RecoveringAudioInputEvent::OpenFailed(error);
                    }
                    self.recovery.failed(now);
                    tracing::warn!(
                        %error,
                        "could not reopen dictation microphone; retrying"
                    );
                }
            }
        }
        if capture_idle {
            self.follow_priority_target(now);
        }
        if (capture_idle || self.input.is_none())
            && self.recovery.should_attempt(now)
            && self.replacement.is_none()
        {
            self.start_replacement();
        }
        if self.recovery.blocks_audio() {
            std::thread::sleep(timeout);
            return RecoveringAudioInputEvent::Timeout;
        }
        let Some(input) = self.input.as_ref() else {
            std::thread::sleep(timeout);
            return RecoveringAudioInputEvent::Timeout;
        };
        match input.recv_timeout(timeout) {
            AudioInputEvent::Chunk {
                samples,
                captured_through,
            } => RecoveringAudioInputEvent::Chunk {
                samples,
                captured_through,
            },
            AudioInputEvent::Timeout => RecoveringAudioInputEvent::Timeout,
            AudioInputEvent::StreamFailed(error) => {
                tracing::warn!(%error, device = %input.device_name, "microphone stream stopped; reopening it");
                self.recovery.request_stream_recovery(Instant::now());
                RecoveringAudioInputEvent::Interrupted
            }
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.input
            .as_ref()
            .expect("microphone sample rate requested while closed")
            .sample_rate
    }

    pub fn device_name(&self) -> &str {
        &self
            .input
            .as_ref()
            .expect("microphone name requested while closed")
            .device_name
    }

    /// Moves an idle warm stream when plugging, unplugging, or a new macOS
    /// default changes which device the priority list resolves to. It reuses
    /// the selection-change path, so the old stream stays until the new one
    /// opens and a recording is never interrupted.
    fn follow_priority_target(&mut self, now: Instant) {
        if self.device_override.is_some()
            || self.replacement.is_some()
            || self.recovery.reason.is_some()
        {
            return;
        }
        let Some(input) = self.input.as_ref() else {
            return;
        };
        if !self.topology.take_change(now) {
            return;
        }
        let host = cpal::default_host();
        let target = match input_candidates(&host, &self.microphone_priority) {
            Ok(mut candidates) => candidates.swap_remove(0).1,
            Err(error) => {
                tracing::debug!(%error, "could not resolve the microphone priority list");
                return;
            }
        };
        if target == input.device_name {
            return;
        }
        tracing::info!(
            from = %input.device_name,
            to = %target,
            "microphone priority target changed; switching"
        );
        self.recovery
            .request_selection_change(self.active_revision, now);
    }

    fn start_replacement(&mut self) {
        let revision = self
            .recovery
            .target_revision
            .unwrap_or(self.active_revision);
        let device_override = self.device_override.clone();
        let microphone_priority = self.microphone_priority.clone();
        let cold_open = self.input.is_none();
        let (sender, receiver) = mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let result =
                open_configured_input(device_override.as_deref(), &microphone_priority, cold_open)
                    .map_err(|error| error.to_string());
            let _ = sender.send((revision, result));
        });
        self.replacement = Some(receiver);
    }

    fn poll_replacement(&mut self) -> Option<InputOpenResult> {
        let result = match self.replacement.as_ref()?.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => (
                self.active_revision,
                Err("microphone replacement worker stopped".into()),
            ),
        };
        self.replacement = None;
        Some(result)
    }
}

/// A cold open falls through the resolved choices so dictation starts on
/// something. A warm switch opens only the resolved target and keeps the old
/// stream on failure, retrying with backoff.
fn open_configured_input(
    device_override: Option<&str>,
    microphone_priority: &[String],
    cold_open: bool,
) -> Result<AudioInput> {
    if let Some(device) = device_override {
        return AudioInput::open_named(device);
    }
    if cold_open {
        AudioInput::open(microphone_priority)
    } else {
        AudioInput::open_prioritized(microphone_priority, false)
    }
}

/// The macOS default input's name, before any Bluetooth avoidance.
pub fn default_input_device_name() -> Option<String> {
    cpal::default_host()
        .default_input_device()
        .and_then(|device| device_name(&device))
}

/// cpal's `Display` for a device errors (so `to_string` panics) once the
/// device disappears, which happens while microphones are being unplugged.
fn device_name(device: &Device) -> Option<String> {
    device
        .description()
        .ok()
        .map(|description| description.name().to_owned())
}

pub fn input_device_names() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let mut names = host
        .input_devices()
        .wrap_err("could not enumerate input devices")?
        .filter_map(|device| device_name(&device))
        .collect::<Vec<_>>();
    names.sort_by_key(|name| name.to_lowercase());
    names.dedup();
    Ok(names)
}

#[cfg(target_os = "linux")]
fn device_matches(device: &Device, query: &str) -> bool {
    device_name(device).is_some_and(|name| device_name_matches(&name, query))
}

/// Priority entries match a device by case-insensitive name fragment.
pub fn device_name_matches(name: &str, query: &str) -> bool {
    let query = query.trim();
    !query.is_empty() && name.to_lowercase().contains(&query.to_lowercase())
}

/// Indices of present devices in priority order, one per entry and without
/// repeats; absent entries are skipped.
fn prioritized_input_indices(priority: &[String], names: &[String]) -> Vec<usize> {
    let mut indices = Vec::new();
    for query in priority {
        if let Some(index) = names
            .iter()
            .position(|name| device_name_matches(name, query))
            && !indices.contains(&index)
        {
            indices.push(index);
        }
    }
    indices
}

/// Present priority devices in order, then the macOS default input. The first
/// candidate is the resolved target; later ones are cold-open fallbacks.
fn input_candidates(
    host: &cpal::Host,
    microphone_priority: &[String],
) -> Result<Vec<(Device, String)>> {
    // A device that vanishes mid-enumeration has no name; skip it.
    let (devices, names): (Vec<_>, Vec<_>) = host
        .input_devices()
        .wrap_err("could not enumerate input devices")?
        .filter_map(|device| device_name(&device).map(|name| (device, name)))
        .unzip();
    let mut candidates = prioritized_input_indices(microphone_priority, &names)
        .into_iter()
        .map(|index| (devices[index].clone(), names[index].clone()))
        .collect::<Vec<_>>();
    if let Some(default) = host.default_input_device() {
        let default = avoid_bluetooth_input(default, &devices);
        if let Some(name) = device_name(&default)
            && !candidates.iter().any(|(_, candidate)| *candidate == name)
        {
            candidates.push((default, name));
        }
    }
    if candidates.is_empty() {
        return Err(eyre!("no prioritized or default input device is available"));
    }
    Ok(candidates)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InputTransport {
    BuiltIn,
    Bluetooth,
    Other,
}

/// Opening a Bluetooth headset's microphone switches it into a low-quality
/// call profile for all audio, so the system-default fallback prefers a
/// built-in mic. An explicitly prioritized Bluetooth device is honored.
#[cfg(target_os = "macos")]
fn automatic_input_index(default: usize, transports: &[InputTransport]) -> usize {
    if transports.get(default) != Some(&InputTransport::Bluetooth) {
        return default;
    }
    transports
        .iter()
        .position(|transport| *transport == InputTransport::BuiltIn)
        .unwrap_or(default)
}

#[cfg(target_os = "macos")]
fn avoid_bluetooth_input(default: Device, devices: &[Device]) -> Device {
    let default_id = default.id().ok();
    let Some(default_index) = devices
        .iter()
        .position(|device| device.id().ok() == default_id)
    else {
        return default;
    };
    let transports = devices.iter().map(input_transport).collect::<Vec<_>>();
    let index = automatic_input_index(default_index, &transports);
    if index != default_index {
        tracing::info!(
            bluetooth = device_name(&devices[default_index]).unwrap_or_default(),
            selected = device_name(&devices[index]).unwrap_or_default(),
            "automatic microphone selection skipped a Bluetooth headset"
        );
    }
    devices[index].clone()
}

#[cfg(target_os = "linux")]
fn avoid_bluetooth_input(default: Device, _devices: &[Device]) -> Device {
    default
}

#[cfg(target_os = "macos")]
fn input_transport(device: &Device) -> InputTransport {
    use std::ffi::c_void;
    use std::ptr::NonNull;

    use objc2_core_audio::{
        AudioObjectGetPropertyData, AudioObjectID, AudioObjectPropertyAddress,
        AudioObjectPropertySelector, kAudioDevicePropertyTransportType,
        kAudioDeviceTransportTypeBluetooth, kAudioDeviceTransportTypeBluetoothLE,
        kAudioDeviceTransportTypeBuiltIn, kAudioHardwarePropertyTranslateUIDToDevice,
        kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal, kAudioObjectSystemObject,
    };
    use objc2_core_foundation::CFString;

    fn read_u32(
        object: AudioObjectID,
        selector: AudioObjectPropertySelector,
        qualifier: Option<NonNull<c_void>>,
        qualifier_size: u32,
    ) -> Option<u32> {
        let mut address = AudioObjectPropertyAddress {
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain,
        };
        let mut value = 0_u32;
        let mut size = std::mem::size_of::<u32>() as u32;
        let status = unsafe {
            AudioObjectGetPropertyData(
                object,
                NonNull::from(&mut address),
                qualifier_size,
                qualifier.map_or(std::ptr::null(), |qualifier| qualifier.as_ptr()),
                NonNull::from(&mut size),
                NonNull::from(&mut value).cast(),
            )
        };
        (status == 0 && size as usize == std::mem::size_of::<u32>()).then_some(value)
    }

    let Ok(id) = device.id() else {
        return InputTransport::Other;
    };
    let uid = CFString::from_str(id.id());
    let uid_ref: *const CFString = &*uid;
    let Some(device_id) = read_u32(
        kAudioObjectSystemObject as AudioObjectID,
        kAudioHardwarePropertyTranslateUIDToDevice,
        Some(NonNull::from(&uid_ref).cast()),
        std::mem::size_of::<*const CFString>() as u32,
    )
    .filter(|device_id| *device_id != 0) else {
        return InputTransport::Other;
    };
    match read_u32(device_id, kAudioDevicePropertyTransportType, None, 0) {
        Some(transport) if transport == kAudioDeviceTransportTypeBuiltIn => InputTransport::BuiltIn,
        Some(transport)
            if transport == kAudioDeviceTransportTypeBluetooth
                || transport == kAudioDeviceTransportTypeBluetoothLE =>
        {
            InputTransport::Bluetooth
        }
        _ => InputTransport::Other,
    }
}

fn mono<T>(samples: &[T], channels: usize, convert: impl Fn(&T) -> f32) -> Vec<f32> {
    samples
        .chunks(channels)
        .map(|frame| frame.iter().map(&convert).sum::<f32>() / channels as f32)
        .collect()
}

fn report_stream_error(sender: &SyncSender<String>, error: String) {
    tracing::error!(%error, "microphone stream failed");
    let _ = sender.try_send(error);
}

fn build_stream<T: SizedSample>(
    device: &Device,
    config: &StreamConfig,
    channels: usize,
    sender: Sender<(Vec<f32>, CaptureInstant)>,
    error_sender: SyncSender<String>,
    convert: impl Fn(T) -> f32 + Send + 'static,
) -> Result<Stream> {
    let sample_rate = config.sample_rate;
    device
        .build_input_stream(
            *config,
            move |data: &[T], info| {
                let captured_through = captured_through(info, data.len() / channels, sample_rate);
                let _ = sender.send((
                    mono(data, channels, |sample| convert(*sample)),
                    captured_through,
                ));
            },
            move |error| report_stream_error(&error_sender, error.to_string()),
            None,
        )
        .wrap_err_with(|| format!("could not open {} microphone stream", T::FORMAT))
}

fn captured_through(
    info: &cpal::InputCallbackInfo,
    frames: usize,
    sample_rate: u32,
) -> CaptureInstant {
    let captured_at = CaptureInstant::from_stream(info.timestamp().capture);
    let duration = duration_for_frames(frames, sample_rate);
    captured_at.checked_add(duration).unwrap_or(captured_at)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_resolves_to_the_first_present_entry() {
        let names = ["MacBook Pro Microphone", "BY-GM18CU", "AirPods Pro"].map(String::from);
        let priority = |entries: &[&str]| {
            entries
                .iter()
                .map(|&entry| entry.to_owned())
                .collect::<Vec<_>>()
        };

        assert_eq!(
            prioritized_input_indices(&priority(&["BY-GM18CU", "MacBook Pro Microphone"]), &names),
            [1, 0]
        );
        // An unplugged entry is skipped until it reappears.
        assert_eq!(
            prioritized_input_indices(&priority(&["Studio Display", "BY-GM18CU"]), &names),
            [1]
        );
        // Case-insensitive fragments still match, without repeating a device.
        assert_eq!(
            prioritized_input_indices(&priority(&["airpods", "AirPods Pro", " "]), &names),
            [2]
        );
        // Nothing present leaves only the system-default fallback.
        assert!(prioritized_input_indices(&priority(&["Studio Display"]), &names).is_empty());
        assert!(prioritized_input_indices(&[], &names).is_empty());
    }

    #[test]
    fn topology_changes_settle_before_re_resolving() {
        let started = Instant::now();
        let mut topology = InputTopology {
            seen: 0,
            changed_at: None,
            polled_at: started,
        };
        if !watch_input_topology() {
            assert!(!topology.take_change(started + Duration::from_secs(1)));
            assert!(topology.take_change(started + INPUT_TOPOLOGY_POLL));
            return;
        }
        topology.seen = INPUT_TOPOLOGY_GENERATION.load(Ordering::Acquire);
        assert!(!topology.take_change(started));
        INPUT_TOPOLOGY_GENERATION.fetch_add(1, Ordering::AcqRel);
        assert!(!topology.take_change(started));
        assert!(!topology.take_change(started + Duration::from_millis(299)));
        assert!(topology.take_change(started + INPUT_TOPOLOGY_SETTLE));
        assert!(!topology.take_change(started + Duration::from_secs(5)));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn system_default_fallback_prefers_built_in_over_a_bluetooth_default() {
        use InputTransport::{Bluetooth, BuiltIn, Other};

        assert_eq!(automatic_input_index(1, &[BuiltIn, Bluetooth]), 0);
        assert_eq!(automatic_input_index(0, &[Bluetooth, Other, BuiltIn]), 2);
        // Non-Bluetooth defaults, such as USB microphones, are kept.
        assert_eq!(automatic_input_index(1, &[BuiltIn, Other]), 1);
        // Without a built-in microphone the Bluetooth default is still used.
        assert_eq!(automatic_input_index(0, &[Bluetooth, Other]), 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mach_event_ticks_are_converted_to_nanoseconds() {
        assert_eq!(scale_mach_ticks(72_000_000, 125, 3), 3_000_000_000);
        assert_eq!(scale_mach_ticks(3_000_000_000, 1, 1), 3_000_000_000);
    }

    #[test]
    fn stream_failures_are_visible_to_the_audio_owner() {
        let (sender, errors) = mpsc::sync_channel(1);

        report_stream_error(&sender, "Device sample rate changed".into());

        assert_eq!(errors.try_recv().unwrap(), "Device sample rate changed");
    }

    #[test]
    fn closed_input_applies_selection_without_opening() {
        let mut input = RecoveringAudioInput::closed(None, 3, &["Old microphone".into()]);

        input.request_selection(4, &["Next microphone".into(), "Old microphone".into()]);

        assert!(!input.is_open());
        assert_eq!(input.active_revision, 4);
        assert_eq!(
            input.microphone_priority,
            ["Next microphone", "Old microphone"]
        );
        assert!(input.replacement.is_none());
    }

    #[test]
    fn failed_startup_retry_keeps_backoff_without_an_open_stream() {
        let mut input = RecoveringAudioInput::closed(None, 3, &["Missing microphone".into()]);
        let started = Instant::now();
        input.finish_initial_open(Err(eyre!("device unavailable")), started);
        assert!(input.is_recovering());
        assert!(
            !input
                .recovery
                .should_attempt(started + Duration::from_millis(249))
        );
        assert!(
            input
                .recovery
                .should_attempt(started + Duration::from_millis(250))
        );
        let (sender, receiver) = mpsc::channel();
        input.replacement = Some(receiver);
        sender.send((3, Err("device unavailable".into()))).unwrap();

        let before = Instant::now();
        assert!(matches!(
            input.recv_timeout(Duration::ZERO, true),
            RecoveringAudioInputEvent::Timeout
        ));
        let after = Instant::now();
        assert!(!input.is_open());
        assert!(!input.is_opening());
        assert!(input.is_recovering());
        let next_attempt = input.recovery.next_attempt.unwrap();
        assert!(next_attempt >= before + Duration::from_millis(500));
        assert!(next_attempt <= after + Duration::from_millis(500));

        input.cancel_open();
        input.request_open();
        assert!(input.is_recovering());
        assert!(!input.is_opening());
        assert_eq!(input.recovery.next_attempt, Some(next_attempt));

        input.request_selection(4, &["Next microphone".into()]);
        assert!(input.is_recovering());
        assert_eq!(input.recovery.target_revision, Some(4));
        assert_eq!(input.active_revision, 3);
        assert_eq!(input.microphone_priority, ["Next microphone"]);
        assert!(input.recovery.should_attempt(Instant::now()));
    }

    #[test]
    fn startup_retry_recovers_metadata_and_audio_without_restarting() {
        let mut input = RecoveringAudioInput::closed(None, 3, &["Test microphone".into()]);
        input.finish_initial_open(Err(eyre!("device unavailable")), Instant::now());
        let (sender, receiver) = mpsc::channel();
        input.replacement = Some(receiver);
        let (opened, samples) = AudioInput::channel_for_test();
        sender.send((3, Ok(opened))).unwrap();

        assert!(matches!(
            input.recv_timeout(Duration::ZERO, true),
            RecoveringAudioInputEvent::Reopened
        ));
        assert!(!input.is_recovering());
        assert!(!input.is_opening());
        assert!(input.recovery.next_attempt.is_none());
        assert_eq!(input.sample_rate(), 48_000);
        assert_eq!(input.device_name(), "Test microphone");
        let at = CaptureInstant::from_nanos(60_000_000_000);
        samples.send((vec![0.25; 480], at)).unwrap();
        assert!(matches!(
            input.recv_timeout(Duration::ZERO, true),
            RecoveringAudioInputEvent::Chunk { samples, captured_through }
                if samples == vec![0.25; 480] && captured_through == at
        ));
    }

    #[test]
    fn startup_recovery_opens_even_when_a_capture_is_pending() {
        let mut input = RecoveringAudioInput::closed(
            Some("HEX nonexistent microphone for pending recovery test"),
            3,
            &[],
        );
        input.request_recovery();
        input.request_selection(4, &["Ignored selection under CLI override".into()]);
        assert_eq!(input.active_revision, 3);
        assert!(input.microphone_priority.is_empty());

        assert!(matches!(
            input.recv_timeout(Duration::ZERO, false),
            RecoveringAudioInputEvent::Timeout
        ));
        assert!(input.is_opening());
        input.cancel_open();
        assert!(input.is_opening());
        input.close();
        assert!(!input.is_opening());
        assert!(!input.is_recovering());
        assert!(input.recovery.next_attempt.is_none());
    }

    #[test]
    fn failed_on_demand_open_does_not_retry_while_idle() {
        let mut input = RecoveringAudioInput::closed(None, 3, &[]);
        let (sender, receiver) = mpsc::channel();
        input.replacement = Some(receiver);
        sender.send((3, Err("device unavailable".into()))).unwrap();

        assert!(matches!(
            input.recv_timeout(Duration::ZERO, true),
            RecoveringAudioInputEvent::OpenFailed(error) if error == "device unavailable"
        ));
        assert!(!input.is_recovering());
        assert!(!input.is_opening());
        assert!(input.recovery.next_attempt.is_none());
    }

    #[test]
    fn microphone_recovery_retries_with_bounded_backoff_and_resets() {
        let started = Instant::now();
        let mut recovery = MicrophoneRecovery::default();

        recovery.request_selection_change(1, started);
        assert!(!recovery.blocks_audio());
        assert!(recovery.should_attempt(started));

        recovery.failed(started);
        recovery.request_selection_change(1, started + Duration::from_millis(100));
        assert!(!recovery.should_attempt(started + Duration::from_millis(249)));
        assert!(recovery.should_attempt(started + Duration::from_millis(250)));

        let mut now = started + Duration::from_millis(250);
        for expected_delay in [500, 1_000, 2_000, 4_000, 5_000, 5_000] {
            recovery.failed(now);
            assert!(!recovery.should_attempt(now + Duration::from_millis(expected_delay - 1)));
            now += Duration::from_millis(expected_delay);
            assert!(recovery.should_attempt(now));
        }

        recovery.recovered();
        assert!(!recovery.blocks_audio());
        assert!(!recovery.should_attempt(now));
        recovery.request_stream_recovery(now);
        assert!(recovery.blocks_audio());
        recovery.failed(now);
        recovery.request_stream_recovery(now + Duration::from_millis(100));
        recovery.request_selection_change(2, now + Duration::from_millis(100));
        assert!(recovery.blocks_audio());
        assert!(recovery.should_attempt(now + Duration::from_millis(100)));
    }
}
