//! Replay a 16 kHz mono WAV through the text-agreement engine (Cohere
//! Transcribe live dictation) in simulated real time: a pass starts once a
//! second of audio, or as soon as the previous pass finishes when it takes
//! longer. Prints each commit with the audio time it appeared at and its lag
//! behind the word's estimated end, then compares the final text with one
//! whole-clip transcription.
//!
//!     cargo run --release --example text_agreement_probe -- model.gguf audio.wav \
//!         [--window 20000] [--interval 1000] [--lag 5000]

#[path = "../src/text_agreement.rs"]
mod text_agreement;

use std::time::Instant;

use text_agreement::{LiveTextPasses, TextAgreement};
use transcribe_cpp::{Backend, Model, ModelOptions, RunOptions, TimestampKind};

const CHUNK_SAMPLES: usize = 35_000 * 16;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect("model path");
    let audio_path = args.next().expect("wav path");
    let mut params = LiveTextPasses::default();
    while let Some(flag) = args.next() {
        let value = args.next().expect("flag value");
        match flag.as_str() {
            "--window" => params.window_ms = value.parse()?,
            "--interval" => params.interval_ms = value.parse()?,
            "--lag" => params.commit_lag_ms = value.parse()?,
            _ => panic!("unknown flag {flag}"),
        }
    }
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
        timestamps: TimestampKind::None,
        language: Some("en".into()),
        ..Default::default()
    };
    let transcribe = |samples: &[f32]| -> Result<(String, u128), Box<dyn std::error::Error>> {
        let started = Instant::now();
        let mut text = Vec::new();
        for chunk in samples.chunks(CHUNK_SAMPLES) {
            let mut chunk = chunk.to_vec();
            chunk.resize(chunk.len().max(16_000), 0.0);
            text.push(model.session()?.run(&chunk, &run)?.text.trim().to_string());
        }
        Ok((text.join(" "), started.elapsed().as_millis()))
    };
    transcribe(&pcm[..16_000])?; // warm up

    let mut agreement = TextAgreement::new(params);
    let mut times = Vec::new();
    let mut lags = Vec::<i64>::new();
    let mut live = Vec::<String>::new();
    let mut end = 16_000usize;
    while end <= pcm.len() {
        let start_ms = agreement.window_start_ms((end / 16) as i64);
        let (text, ms) = transcribe(&pcm[start_ms as usize * 16..end])?;
        times.push((ms, (end / 16) as i64 - start_ms));
        let at_ms = (end / 16) as i64;
        let committed = agreement.pass(&text, start_ms, at_ms);
        live.extend(committed.iter().map(|w| w.text.clone()));
        if !committed.is_empty() {
            let words: Vec<_> = committed.iter().map(|w| w.text.as_str()).collect();
            let lag: Vec<i64> = committed.iter().map(|w| at_ms - w.end_ms).collect();
            lags.extend(&lag);
            println!(
                "audio {:>5} ms (window {:>4.1} s, pass {ms:>3} ms, lag {}-{} ms): {}",
                at_ms,
                (at_ms - start_ms) as f64 / 1000.0,
                lag.iter().min().unwrap(),
                lag.iter().max().unwrap(),
                words.join(" ")
            );
        } else if std::env::var_os("PROBE_VERBOSE").is_some() {
            let pasted = live[live.len().saturating_sub(8)..].join(" ");
            println!(
                "audio {at_ms:>5} ms (window {:>4.1} s): NOTHING | pasted tail: …{pasted} | pass: {text}",
                (at_ms - start_ms) as f64 / 1000.0,
            );
        }
        // The next pass starts on the interval, or right after a slow pass.
        end += (ms.max(u128::from(params.interval_ms)) as usize) * 16;
    }
    let mut tail = pcm.clone();
    tail.extend_from_slice(&[0.0; 3_200]);
    let start_ms = agreement.final_start_ms((pcm.len() / 16) as i64);
    let (text, ms) = transcribe(&tail[start_ms as usize * 16..])?;
    let rest: Vec<_> = agreement
        .finish(&text, start_ms, (tail.len() / 16) as i64)
        .into_iter()
        .map(|w| w.text)
        .collect();
    println!(
        "release: final pass over {:.1} s took {ms} ms, pasted: {}",
        (tail.len() / 16) as f64 / 1000.0 - start_ms as f64 / 1000.0,
        rest.join(" ")
    );
    live.extend(rest.iter().cloned());
    let live = live.join(" ");
    let (whole, whole_ms) = transcribe(&tail)?;
    times.sort();
    lags.sort();
    let window_times: Vec<_> = times.iter().filter(|(_, w)| *w >= 19_000).collect();
    println!(
        "{} passes, median {} ms, max {} ms; full 20 s windows: {} passes, median {} ms",
        times.len(),
        times[times.len() / 2].0,
        times[times.len() - 1].0,
        window_times.len(),
        window_times.get(window_times.len() / 2).map_or(0, |t| t.0),
    );
    if !lags.is_empty() {
        println!(
            "commit lag behind estimated word end: min {} ms, median {} ms, max {} ms",
            lags[0],
            lags[lags.len() / 2],
            lags[lags.len() - 1]
        );
    }
    println!("\nlive:  {live}\nwhole ({whole_ms} ms): {whole}");
    let (a, b) = (keys(&live), keys(&whole));
    let same = lcs(&a, &b);
    println!(
        "\nwords: live {}, whole {}, in common {} ({} only live, {} only whole){}",
        a.len(),
        b.len(),
        same,
        a.len() - same,
        b.len() - same,
        if a == b { " -- identical" } else { "" }
    );
    Ok(())
}

fn keys(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

fn lcs(a: &[String], b: &[String]) -> usize {
    let mut row = vec![0; b.len() + 1];
    for x in a {
        let mut prev = 0;
        for (j, y) in b.iter().enumerate() {
            let here = row[j + 1];
            row[j + 1] = if x == y {
                prev + 1
            } else {
                row[j + 1].max(row[j])
            };
            prev = here;
        }
    }
    row[b.len()]
}
