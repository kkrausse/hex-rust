//! Live dictation for models that give neither streaming nor word times
//! (Cohere Transcribe). The trailing window is transcribed about once a
//! second; each pass's text is aligned against what is already pasted, and a
//! word after that point commits once two consecutive passes agree on it and
//! it is estimated to end at least `commit_lag_ms` before the newest audio.
//! The model gives no word times, so a word's time is estimated by spreading
//! the pass's text evenly over the window by character position.
//! Pure: the caller runs the passes and feeds their text and window in.

use serde::{Deserialize, Serialize};

/// Each pass reaches at least this far back into pasted text so it can be
/// aligned against it.
const OVERLAP_MS: i64 = 2_500;
/// Hard cap on one pass, under Cohere's 35 s chunk.
const MAX_PASS_MS: i64 = 30_000;
/// Pasted words an alignment looks at.
const ALIGN_TAIL_WORDS: usize = 16;

/// The user-tunable part of the rule, stored in settings.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct LiveTextPasses {
    /// Trailing audio each pass transcribes.
    pub window_ms: u32,
    /// A pass starts at most this often; a slow pass is never queued behind.
    pub interval_ms: u32,
    /// Only text estimated this far behind the newest audio commits.
    pub commit_lag_ms: u32,
}

impl Default for LiveTextPasses {
    fn default() -> Self {
        Self {
            window_ms: 20_000,
            interval_ms: 1_000,
            commit_lag_ms: 5_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CommittedWord {
    pub text: String,
    /// Estimated end of the word, in ms from the start of capture.
    pub end_ms: i64,
}

struct PassWord {
    text: String,
    key: String,
    end_ms: i64,
}

#[derive(Default)]
pub struct TextAgreement {
    params: LiveTextPasses,
    committed: Vec<CommittedWord>,
    /// The previous pass's words after the committed point.
    previous: Vec<String>,
    committed_end_ms: i64,
    /// Passes in a row that could not be aligned. The estimated end of the
    /// pasted text can sit seconds late (word times are spread evenly, so a
    /// pause throws them off); each miss reaches one more overlap further
    /// back so the next pass contains pasted words again.
    misses: i64,
}

impl TextAgreement {
    pub fn new(params: LiveTextPasses) -> Self {
        Self {
            params,
            ..Self::default()
        }
    }

    /// Where the next pass should begin, in ms, once `end_ms` of audio has
    /// been captured: the trailing window, reaching back into pasted text
    /// when commits lag so the pass can be aligned, capped at `MAX_PASS_MS`.
    pub fn window_start_ms(&self, end_ms: i64) -> i64 {
        let window = i64::from(self.params.window_ms).min(MAX_PASS_MS);
        (end_ms - window)
            .min(self.committed_end_ms - self.overlap_ms())
            .max(end_ms - MAX_PASS_MS)
            .max(0)
    }

    /// Where the final pass at release should begin: like `window_start_ms`
    /// but uncapped, so it always covers everything not yet pasted. The
    /// caller chunks it if it exceeds the model's input limit.
    pub fn final_start_ms(&self, end_ms: i64) -> i64 {
        (end_ms - i64::from(self.params.window_ms))
            .min(self.committed_end_ms - self.overlap_ms())
            .max(0)
    }

    fn overlap_ms(&self) -> i64 {
        OVERLAP_MS * (1 + self.misses)
    }

    /// One pass's `text` over the audio from `start_ms` to `end_ms`, in ms
    /// from the start of capture. Returns the words it commits.
    pub fn pass(&mut self, text: &str, start_ms: i64, end_ms: i64) -> Vec<CommittedWord> {
        let words = timed_words(text, start_ms, end_ms);
        let Some(aligned) = self.align(&words, start_ms, false) else {
            tracing::debug!(start_ms, end_ms, "text pass did not align; nothing commits");
            self.previous.clear();
            self.misses += 1;
            return Vec::new();
        };
        self.misses = 0;
        // This pass's estimate of where the pasted text ends replaces the
        // older one, so an estimate thrown late by a pause does not persist.
        if aligned > 0 {
            self.committed_end_ms = words[aligned - 1].end_ms;
        }
        let pending = &words[aligned..];
        let lag = i64::from(self.params.commit_lag_ms);
        let ready = pending
            .iter()
            .enumerate()
            .take_while(|(index, word)| {
                word.end_ms <= end_ms - lag && self.previous.get(*index) == Some(&word.key)
            })
            .count();
        let newly = self.commit(&pending[..ready]);
        self.previous = pending[ready..]
            .iter()
            .map(|word| word.key.clone())
            .collect();
        newly
    }

    /// The final pass at release: everything after the committed point
    /// commits. If alignment fails, the cut falls where the pasted text is
    /// estimated to end.
    pub fn finish(&mut self, text: &str, start_ms: i64, end_ms: i64) -> Vec<CommittedWord> {
        let words = timed_words(text, start_ms, end_ms);
        let aligned = self
            .align(&words, start_ms, true)
            .unwrap_or_else(|| expected_split(&words, self.committed_end_ms));
        self.previous.clear();
        self.commit(&words[aligned..])
    }

    fn commit(&mut self, words: &[PassWord]) -> Vec<CommittedWord> {
        let newly: Vec<CommittedWord> = words
            .iter()
            .map(|word| CommittedWord {
                text: word.text.clone(),
                end_ms: word.end_ms,
            })
            .collect();
        if let Some(last) = newly.last() {
            self.committed_end_ms = self.committed_end_ms.max(last.end_ms);
        }
        self.committed.extend(newly.iter().cloned());
        newly
    }

    /// The index of the first pass word after the pasted text, or `None`
    /// when the pass cannot be placed against it. The pasted tail is aligned
    /// backwards from each possible split with fuzzy matches and gaps, so a
    /// first word cut mid-word or a small edit does not break it. Repetitive
    /// speech ("it was the ... it was the ...") matches in several places,
    /// so each word of distance from where the pasted text is estimated to
    /// end costs half a point.
    fn align(&self, words: &[PassWord], start_ms: i64, lenient: bool) -> Option<usize> {
        if self.committed.is_empty() {
            return Some(0);
        }
        // The pass starts after all pasted audio (only a live pass capped at
        // MAX_PASS_MS can): words between would be lost, so wait for release.
        if self.committed_end_ms <= start_ms {
            return lenient.then_some(0);
        }
        let tail: Vec<String> = self.committed
            [self.committed.len().saturating_sub(ALIGN_TAIL_WORDS)..]
            .iter()
            .map(|word| key(&word.text))
            .collect();
        let expected = expected_split(words, self.committed_end_ms) as i64;
        let needed = if lenient || tail.len() == 1 { 2 } else { 3 };
        let mut best: Option<(i64, usize)> = None;
        for split in 1..=words.len() {
            let score = suffix_score(&tail, &words[..split]);
            if score < needed {
                continue;
            }
            let score = 2 * i64::from(score) - (split as i64 - expected).abs();
            // On a tie the later split wins: a re-heard last word ("the" →
            // "a") is far likelier than a dropped one, and pasting it twice
            // is worse.
            if best.is_none_or(|(best, _)| score >= best) {
                best = Some((score, split));
            }
        }
        best.map(|(_, split)| split)
    }
}

/// Best score of aligning the end of `tail` with the end of `words`, both
/// read backwards, stopping anywhere: +2 per match, +1 per near match, -1
/// per mismatch or skipped pass word, -2 per skipped pasted word. Pasted
/// words were agreed by two passes, so a pass is likelier to re-hear one
/// ("the" → "a") than to lose it.
fn suffix_score(tail: &[String], words: &[PassWord]) -> i32 {
    let a: Vec<&str> = tail.iter().rev().map(String::as_str).collect();
    let b: Vec<&str> = words
        .iter()
        .rev()
        .take(tail.len() + 4)
        .map(|word| word.key.as_str())
        .collect();
    let mut row: Vec<i32> = (0..=b.len() as i32).map(|j| -j).collect();
    let mut best = 0;
    for (i, x) in a.iter().enumerate() {
        let mut next = vec![-2 * (i as i32 + 1); b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            let pair = if x == y {
                2
            } else if near(x, y) {
                1
            } else {
                -1
            };
            next[j + 1] = (row[j] + pair).max(row[j + 1] - 2).max(next[j] - 1);
            best = best.max(next[j + 1]);
        }
        row = next;
    }
    best
}

/// One edit apart, or `b` is the end of `a` (a word cut at the window start).
fn near(a: &str, b: &str) -> bool {
    if b.len() >= 3 && a.ends_with(b) {
        return true;
    }
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    if a.len() < 4 || b.len() < 4 || a.len().abs_diff(b.len()) > 1 {
        return false;
    }
    // Edit distance of at most one.
    let (mut i, mut j, mut edits) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            i += 1;
            j += 1;
            continue;
        }
        edits += 1;
        if edits > 1 {
            return false;
        }
        match a.len().cmp(&b.len()) {
            std::cmp::Ordering::Greater => i += 1,
            std::cmp::Ordering::Less => j += 1,
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    edits + (a.len() - i) + (b.len() - j) <= 1
}

/// The first word estimated to end after `end_ms`.
fn expected_split(words: &[PassWord], end_ms: i64) -> usize {
    words
        .iter()
        .position(|word| word.end_ms > end_ms)
        .unwrap_or(words.len())
}

/// Split a pass's text into words, folding punctuation-only tokens into the
/// word before, and give each an end time spread evenly over the window by
/// character position.
fn timed_words(text: &str, start_ms: i64, end_ms: i64) -> Vec<PassWord> {
    let mut words: Vec<(String, String)> = Vec::new();
    for token in text.split_whitespace() {
        let token_key = key(token);
        match words.last_mut() {
            Some((text, _)) if token_key.is_empty() => {
                text.push(' ');
                text.push_str(token);
            }
            _ if token_key.is_empty() => {}
            _ => words.push((token.to_string(), token_key)),
        }
    }
    let total: usize = words
        .iter()
        .map(|(_, key)| key.chars().count().max(1))
        .sum();
    let mut through = 0;
    words
        .into_iter()
        .map(|(text, key)| {
            through += key.chars().count().max(1);
            PassWord {
                end_ms: start_ms + (end_ms - start_ms) * through as i64 / total.max(1) as i64,
                text,
                key,
            }
        })
        .collect()
}

fn key(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}
