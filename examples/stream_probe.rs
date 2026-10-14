//! Feed a 16 kHz mono WAV through a transcribe.cpp stream in 100 ms chunks and
//! print every committed/tentative change with the audio time it appeared at.
//!
//!     cargo run --example stream_probe -- model.gguf audio.wav [auto|stable]

use std::time::Instant;

use transcribe_cpp::{
    Backend, CommitPolicy, Model, ModelOptions, RunOptions, StreamOptions, TimestampKind,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect("model path");
    let audio_path = args.next().expect("wav path");
    let policy = CommitPolicy::Auto;
    let chunk_ms = args.next().and_then(|v| v.parse().ok());
    let right_ms = args.next().and_then(|v| v.parse().ok());
    let mut reader = hound::WavReader::open(&audio_path)?;
    let pcm: Vec<f32> = reader
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
    let caps = model.capabilities();
    println!(
        "arch={} variant={} supports_streaming={}",
        model.arch(),
        model.variant(),
        caps.supports_streaming
    );
    let mut session = model.session()?;
    let run = RunOptions {
        timestamps: TimestampKind::None,
        ..Default::default()
    };
    let opts = StreamOptions {
        commit_policy: policy,
        family: Some(transcribe_cpp::StreamExtension::ParakeetBuffered(
            transcribe_cpp::ParakeetBufferedStreamOptions {
                left_ms: None,
                chunk_ms,
                right_ms,
            },
        )),
        ..Default::default()
    };
    for pass in 0..2 {
        let mut stream = session.stream(&run, &opts)?;
        let started = Instant::now();
        let mut feed_ms = 0u128;
        for (i, frame) in pcm.chunks(1600).enumerate() {
            let t = Instant::now();
            let update = stream.feed(frame)?;
            feed_ms = feed_ms.max(t.elapsed().as_millis());
            if pass == 1 && update.committed_changed {
                let text = stream.text();
                println!(
                    "audio {:>5} ms | committed: {:?} | tentative: {:?}",
                    (i + 1) * 100,
                    text.committed,
                    text.tentative
                );
            }
        }
        let t = Instant::now();
        stream.finalize()?;
        println!(
            "pass {pass}: total {} ms, worst feed {feed_ms} ms, finalize {} ms\nfinal: {:?}",
            started.elapsed().as_millis(),
            t.elapsed().as_millis(),
            stream.text().committed
        );
    }
    Ok(())
}
