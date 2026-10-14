//! Replay a 16 kHz mono WAV through the agreement engine on a virtual clock:
//! a pass every 300 ms of audio, as live dictation runs it. Prints each commit
//! with the audio time it appeared at, and the final pass time after release.
//!
//!     cargo run --release --example agreement_probe -- model.gguf audio.wav

#[path = "../src/agreement.rs"]
mod agreement;

use std::time::Instant;

use agreement::{Agreement, TimedWord};
use transcribe_cpp::{Backend, Model, ModelOptions, RunOptions, TimestampKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect("model path");
    let audio_path = args.next().expect("wav path");
    let pcm: Vec<f32> = hound::WavReader::open(&audio_path)?
        .samples::<i16>()
        .map(|s| s.map(|s| f32::from(s) / 32768.0))
        .collect::<Result<_, _>>()?;
    let model = Model::load_with(
        &model_path,
        &ModelOptions {
            backend: Backend::Metal,
            device: None,
        },
    )?;
    let run = RunOptions {
        timestamps: TimestampKind::Word,
        ..Default::default()
    };
    let mut pass = |samples: &[f32],
                    offset_ms: i64|
     -> Result<(Vec<TimedWord>, u128), Box<dyn std::error::Error>> {
        let mut samples = samples.to_vec();
        samples.resize(samples.len().max(16_000), 0.0);
        let started = Instant::now();
        let transcript = model.session()?.run(&samples, &run)?;
        let words = transcript
            .words
            .into_iter()
            .filter(|w| !w.text.trim().is_empty())
            .map(|w| TimedWord {
                text: w.text.trim().to_string(),
                start_ms: w.t0_ms + offset_ms,
                end_ms: w.t1_ms + offset_ms,
            })
            .collect();
        Ok((words, started.elapsed().as_millis()))
    };
    pass(&pcm[..16_000], 0)?; // warm up
    let mut agreement = Agreement::default();
    let mut times = Vec::new();
    for end in (4_000..=pcm.len()).step_by(4_800) {
        let start = (agreement.window_start_ms() as usize * 16).min(end);
        let (words, ms) = pass(&pcm[start..end], agreement.window_start_ms())?;
        times.push(ms);
        let committed = agreement.pass(words, (end / 16) as i64);
        if !committed.is_empty() {
            let texts: Vec<_> = committed.iter().map(|w| w.text.as_str()).collect();
            println!(
                "audio {:>5} ms (window {:>4.1} s, pass {ms} ms): {}",
                end / 16,
                (end - start) as f64 / 16_000.0,
                texts.join(" ")
            );
        }
    }
    let mut tail = pcm[(agreement.window_start_ms() as usize * 16).min(pcm.len())..].to_vec();
    tail.extend_from_slice(&[0.0; 3_200]);
    let (words, ms) = pass(&tail, agreement.window_start_ms())?;
    let rest: Vec<_> = agreement
        .finish(words)
        .into_iter()
        .map(|w| w.text)
        .collect();
    times.sort();
    println!(
        "release: final pass over {:.1} s took {ms} ms, pasted: {}",
        tail.len() as f64 / 16_000.0,
        rest.join(" ")
    );
    println!(
        "{} passes, median {} ms, max {} ms",
        times.len(),
        times[times.len() / 2],
        times[times.len() - 1]
    );
    Ok(())
}
