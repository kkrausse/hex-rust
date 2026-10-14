//! Live dictation for models without native streaming, ported from
//! kkrausse/parakeet-web `lib/src/stream.ts` (rule: docs/streaming-engine.md
//! there). The trailing window is transcribed again and again; a word is
//! committed once consecutive passes agree on it and enough audio follows it.
//! Pure: the caller runs the passes and feeds their words in.

use serde::{Deserialize, Serialize};

/// Two passes place the same word within this distance.
const SAME_PLACE_MS: i64 = 240;
/// Consecutive passes that must agree on a word; the hold usually decides.
const MIN_PASSES: u32 = 2;
/// A closing full stop can stretch a word across a pause; cap its length.
const MAX_WORD_MS: i64 = 1_200;
/// Window cap beyond the kept context: a pass over context plus this much
/// audio commits everything older than `FULL_TAIL_MS` before its end.
const UNCOMMITTED_MAX_MS: i64 = 8_000;
const FULL_TAIL_MS: i64 = 1_000;
/// Words carry times on an 80 ms grid.
const GRID_MS: i64 = 80;

/// The user-tunable part of the rule, stored in settings.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct LivePasses {
    /// A pass starts at most this often.
    pub interval_ms: u32,
    /// Audio kept before the committed point when the window moves.
    pub context_ms: u32,
    /// Audio that must follow a word's end before it commits.
    pub hold_ms: u32,
}

impl Default for LivePasses {
    fn default() -> Self {
        Self {
            interval_ms: 300,
            context_ms: 4_000,
            hold_ms: 1_200,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TimedWord {
    pub text: String,
    pub start_ms: i64,
    pub end_ms: i64,
}

struct Candidate {
    word: TimedWord,
    passes: u32,
}

#[derive(Default)]
pub struct Agreement {
    params: LivePasses,
    committed: Vec<TimedWord>,
    candidates: Vec<Candidate>,
    window_start_ms: i64,
}

impl Agreement {
    pub fn new(params: LivePasses) -> Self {
        Self {
            params,
            ..Self::default()
        }
    }

    /// Where the next pass's audio starts, in ms from the start of capture.
    pub fn window_start_ms(&self) -> i64 {
        self.window_start_ms
    }

    pub fn window_full(&self, audio_ms: i64) -> bool {
        audio_ms - self.window_start_ms >= self.max_window_ms()
    }

    /// Take one pass over `[window_start_ms, audio_ms)`; word times are
    /// absolute. Returns the words committed by it.
    pub fn pass(&mut self, words: Vec<TimedWord>, audio_ms: i64) -> Vec<TimedWord> {
        let fresh = self.after_committed(words);
        let mut agreed = 0;
        for (candidate, word) in self.candidates.iter_mut().zip(&fresh) {
            if normalized(&candidate.word.text) != normalized(&word.text)
                || (candidate.word.start_ms - word.start_ms).abs() > SAME_PLACE_MS
            {
                break;
            }
            candidate.passes += 1;
            candidate.word = word.clone();
            agreed += 1;
        }
        self.candidates.truncate(agreed);
        self.candidates.extend(
            fresh
                .into_iter()
                .skip(agreed)
                .map(|word| Candidate { word, passes: 1 }),
        );

        let full = self.window_full(audio_ms);
        let ready = self
            .candidates
            .iter()
            .take_while(|candidate| {
                if full {
                    return candidate.word.start_ms < audio_ms - FULL_TAIL_MS;
                }
                let end = candidate
                    .word
                    .end_ms
                    .min(candidate.word.start_ms + MAX_WORD_MS);
                candidate.passes >= MIN_PASSES && audio_ms - end >= i64::from(self.params.hold_ms)
            })
            .count();
        let newly = self.commit(ready);
        if full && newly.is_empty() {
            // Nothing to cut at: drop the oldest audio rather than stall.
            self.window_start_ms = audio_ms - self.max_window_ms() + FULL_TAIL_MS;
        }
        newly
    }

    fn max_window_ms(&self) -> i64 {
        i64::from(self.params.context_ms) + UNCOMMITTED_MAX_MS
    }

    /// Final pass at release: everything after the committed point commits.
    pub fn finish(&mut self, words: Vec<TimedWord>) -> Vec<TimedWord> {
        self.candidates = self
            .after_committed(words)
            .into_iter()
            .map(|word| Candidate { word, passes: 1 })
            .collect();
        self.commit(self.candidates.len())
    }

    /// Drop the words of a pass that lie at or before the committed point.
    /// The split is found by time, then corrected by matching the last
    /// committed word, which can sit a frame earlier or later each pass.
    fn after_committed(&self, words: Vec<TimedWord>) -> Vec<TimedWord> {
        let Some(last) = self.committed.last() else {
            return words;
        };
        // A moved window can shift a word by more than a frame, so match
        // generously; by time, anything starting before the end of the last
        // committed word is that word or older.
        let by_match = words.iter().rposition(|word| {
            normalized(&word.text) == normalized(&last.text)
                && (word.start_ms - last.start_ms).abs() <= 2 * SAME_PLACE_MS
        });
        let split = match by_match {
            Some(index) => index + 1,
            None => words
                .iter()
                .position(|word| word.start_ms >= last.end_ms - GRID_MS)
                .unwrap_or(words.len()),
        };
        words.into_iter().skip(split).collect()
    }

    fn commit(&mut self, count: usize) -> Vec<TimedWord> {
        let newly: Vec<TimedWord> = self
            .candidates
            .drain(..count)
            .map(|candidate| candidate.word)
            .collect();
        self.committed.extend(newly.iter().cloned());
        if !newly.is_empty() {
            self.move_window();
        }
        newly
    }

    /// Start the window up to `context_ms` before the committed point, in the
    /// widest gap between committed words there, so no pass starts mid-word.
    fn move_window(&mut self) {
        let Some(last) = self.committed.last() else {
            return;
        };
        let earliest = last.end_ms - i64::from(self.params.context_ms);
        let gap = self
            .committed
            .windows(2)
            .filter(|pair| pair[0].end_ms >= earliest)
            .max_by_key(|pair| pair[1].start_ms - pair[0].end_ms)
            .map(|pair| (pair[0].end_ms + pair[1].start_ms) / 2)
            // No gap that recent: start at a committed word's start instead.
            .or_else(|| {
                self.committed
                    .iter()
                    .find(|word| word.start_ms >= earliest)
                    .map(|word| word.start_ms - GRID_MS)
            });
        if let Some(start) = gap {
            let start = start / GRID_MS * GRID_MS;
            self.window_start_ms = self.window_start_ms.max(start);
        }
    }
}

fn normalized(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(spec: &[(&str, i64)]) -> Vec<TimedWord> {
        spec.iter()
            .map(|&(text, start_ms)| TimedWord {
                text: text.into(),
                start_ms,
                end_ms: start_ms + 300,
            })
            .collect()
    }

    fn texts(words: &[TimedWord]) -> Vec<&str> {
        words.iter().map(|word| word.text.as_str()).collect()
    }

    #[test]
    fn commits_only_words_passes_agree_on_after_the_hold() {
        let mut agreement = Agreement::default();
        assert!(
            agreement
                .pass(words(&[("hello", 0), ("world", 900)]), 1_000)
                .is_empty()
        );
        // Same words a frame later, but too little audio after them yet.
        assert!(
            agreement
                .pass(words(&[("Hello,", 80), ("world", 900)]), 1_250)
                .is_empty()
        );
        let spoken = words(&[("hello", 0), ("world", 900), ("again", 1_500)]);
        assert_eq!(texts(&agreement.pass(spoken.clone(), 2_000)), ["hello"]);
        // "world" ends at 1.2 s, so it waits for 2.4 s of audio.
        assert_eq!(texts(&agreement.pass(spoken, 2_400)), ["world"]);
    }

    #[test]
    fn a_changed_word_restarts_agreement_from_there() {
        let mut agreement = Agreement::default();
        agreement.pass(words(&[("I", 0), ("scream", 400)]), 1_000);
        agreement.pass(words(&[("ice", 0), ("cream", 400)]), 1_600);
        let committed = agreement.pass(words(&[("ice", 0), ("cream", 400)]), 2_200);
        assert_eq!(texts(&committed), ["ice", "cream"]);
    }

    #[test]
    fn a_shifted_last_word_is_not_committed_twice() {
        let mut agreement = Agreement::default();
        agreement.pass(words(&[("send", 0), ("numbers", 400)]), 1_000);
        agreement.pass(words(&[("send", 0), ("numbers", 400)]), 1_500);
        agreement.pass(words(&[("send", 0), ("numbers", 400)]), 2_000);
        // After the window moved, "numbers" comes back 400 ms later.
        let rest = agreement.finish(words(&[("numbers", 800), ("today", 1_200)]));
        assert_eq!(texts(&rest), ["today"]);
    }

    #[test]
    fn finish_commits_the_rest_after_the_committed_point() {
        let mut agreement = Agreement::default();
        agreement.pass(words(&[("one", 0), ("two", 400)]), 1_000);
        agreement.pass(words(&[("one", 0), ("two", 400)]), 1_500);
        agreement.pass(words(&[("one", 0), ("two", 400)]), 2_000);
        let rest = agreement.finish(words(&[("one", 80), ("two", 400), ("three", 2_100)]));
        assert_eq!(texts(&rest), ["three"]);
    }

    #[test]
    fn the_window_moves_into_a_gap_and_never_back() {
        let mut agreement = Agreement::default();
        let spoken = words(&[("a", 0), ("b", 400), ("c", 3_000), ("d", 6_000)]);
        agreement.pass(spoken.clone(), 7_000);
        agreement.pass(spoken.clone(), 7_500);
        agreement.pass(spoken, 8_000);
        // Widest gap within 4 s of d's end: c (ends 3.3 s) to d (6 s).
        assert_eq!(agreement.window_start_ms(), 4_640);
    }

    #[test]
    fn a_full_window_commits_all_but_the_last_second() {
        let mut agreement = Agreement::default();
        let committed = agreement.pass(words(&[("x", 0), ("y", 5_000), ("z", 11_500)]), 12_000);
        assert_eq!(texts(&committed), ["x", "y"]);
        // y is the newest committed word within 4 s, so the window starts there.
        assert_eq!(agreement.window_start_ms(), 4_880);
        assert!(!agreement.window_full(12_000));
    }
}
