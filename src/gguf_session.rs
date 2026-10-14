//! The `transcribe-cpp` session owner shared by the macOS and Linux GGUF
//! transcribers. It recreates the native session around every run so retained
//! scheduler scratch never outlives the inference that sized it.

use std::path::Path;

use color_eyre::eyre::{Result, eyre};

use crate::transcription_models::{
    AUTO_LANGUAGE, ModelDefinition, ModelRuntime, TranscriptionSelection,
};

pub(crate) fn validate_gguf_artifact(
    path: &Path,
    definition: &ModelDefinition,
    architecture: &str,
    variant: &str,
    selection: &TranscriptionSelection,
    supports_language_detect: bool,
) -> Result<Option<String>> {
    let ModelRuntime::Gguf(artifact) = definition.runtime else {
        return Err(eyre!(
            "{} is not a GGUF transcription model",
            definition.name
        ));
    };
    if architecture != artifact.architecture || variant != artifact.variant {
        return Err(eyre!(
            "{} contains {architecture}/{variant}, expected {}/{}",
            path.display(),
            artifact.architecture,
            artifact.variant
        ));
    }
    if selection.language == AUTO_LANGUAGE && !supports_language_detect {
        return Err(eyre!(
            "{} does not advertise automatic language detection",
            definition.name
        ));
    }
    Ok(definition
        .runtime_language_hint(&selection.language)
        .map(str::to_string))
}

pub(crate) struct OfflineGgufSession {
    model: transcribe_cpp::Model,
    session: Option<transcribe_cpp::Session>,
}

impl OfflineGgufSession {
    pub(crate) fn new(model: transcribe_cpp::Model) -> transcribe_cpp::Result<Self> {
        let session = model.session()?;
        Ok(Self {
            model,
            session: Some(session),
        })
    }

    pub(crate) fn run(
        &mut self,
        samples: &[f32],
        options: &transcribe_cpp::RunOptions,
    ) -> transcribe_cpp::Result<transcribe_cpp::Transcript> {
        // transcribe-cpp 0.1.x retains input-sized scheduler buffers for the
        // session lifetime. Drop each used session before creating its
        // successor so a long inference cannot pin that high-water mark or
        // overlap two sessions' persistent decoder state.
        let mut session = match self.session.take() {
            Some(session) => session,
            None => self.model.session()?,
        };
        let result = session.run(samples, options);
        drop(session);
        match self.model.session() {
            Ok(session) => {
                self.session = Some(session);
                result
            }
            Err(recovery_error) => match result {
                Ok(_) => Err(recovery_error),
                Err(run_error) => Err(run_error),
            },
        }
    }

    pub(crate) fn supports_streaming(&self) -> bool {
        self.model.capabilities().supports_streaming
    }

    /// Run `live` against one stream, then replace the session the same way
    /// `run` does so stream scratch never outlives the dictation.
    pub(crate) fn stream<T>(
        &mut self,
        options: &transcribe_cpp::RunOptions,
        live: impl FnOnce(&mut transcribe_cpp::Stream<'_>) -> T,
    ) -> transcribe_cpp::Result<T> {
        let mut session = match self.session.take() {
            Some(session) => session,
            None => self.model.session()?,
        };
        let result = session
            .stream(options, &transcribe_cpp::StreamOptions::default())
            .map(|mut stream| live(&mut stream));
        drop(session);
        let session = self.model.session()?;
        self.session = Some(session);
        result
    }
}
