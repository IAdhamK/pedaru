//! Translation API client for local LM Studio
//!
//! This module provides functionality to translate text using a local
//! OpenAI-compatible endpoint (LM Studio by default).

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;

use crate::error::{GeminiError, PedaruError};

/// Default LM Studio OpenAI-compatible API base URL
const DEFAULT_LM_STUDIO_API_BASE: &str = "http://127.0.0.1:1234/v1";
const DEFAULT_TIMEOUT_SECONDS: u64 = 180;
const DEFAULT_MAX_TOKENS: u32 = 120;
const MAX_CONTEXT_CHARS: usize = 800;

fn env_u64(name: &str, default: u64, min: u64, max: u64) -> u64 {
    match env::var(name) {
        Ok(raw) => match raw.parse::<u64>() {
            Ok(v) => v.clamp(min, max),
            Err(_) => default,
        },
        Err(_) => default,
    }
}

fn env_u32(name: &str, default: u32, min: u32, max: u32) -> u32 {
    match env::var(name) {
        Ok(raw) => match raw.parse::<u32>() {
            Ok(v) => v.clamp(min, max),
            Err(_) => default,
        },
        Err(_) => default,
    }
}

fn clamp_context(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

fn value_to_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Value::Array(items) => {
            // Join useful string-ish parts from array content
            let parts: Vec<String> = items.iter().filter_map(value_to_text).collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n"))
            }
        }
        Value::Object(_) => {
            // Common keys used by local OpenAI-compatible servers/models
            for key in [
                "translation",
                "translated_text",
                "summary",
                "text",
                "content",
                "output",
                "response",
                "answer",
                "result",
                "message",
            ] {
                if let Some(v) = value.get(key) {
                    if let Some(s) = value_to_text(v) {
                        return Some(s);
                    }
                }
            }

            // Handle completion-like payload nested inside JSON
            value
                .get("choices")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.first())
                .and_then(|first| first.get("message"))
                .and_then(|message| message.get("content"))
                .and_then(value_to_text)
        }
        _ => None,
    }
}

fn extract_points(value: &Value) -> Vec<String> {
    if let Some(points_array) = value.get("points").and_then(|v| v.as_array()) {
        return points_array
            .iter()
            .filter_map(value_to_text)
            .collect::<Vec<String>>();
    }
    if let Some(single_point) = value.get("points").and_then(value_to_text) {
        return vec![single_point];
    }
    Vec::new()
}

fn extract_chat_choice_text(choice: &ChatChoice) -> Option<String> {
    if let Some(message) = &choice.message {
        if let Some(s) = value_to_text(&message.content) {
            let cleaned = clean_model_output(&s);
            if !cleaned.trim().is_empty() {
                return Some(cleaned);
            }
        }

        if let Some(s) = value_to_text(&message.reasoning_content) {
            let cleaned = clean_model_output(&s);
            if !cleaned.trim().is_empty() {
                return Some(cleaned);
            }
        }

        if let Some(s) = value_to_text(&message.reasoning) {
            let cleaned = clean_model_output(&s);
            if !cleaned.trim().is_empty() {
                return Some(cleaned);
            }
        }
    }

    choice
        .text
        .as_deref()
        .map(clean_model_output)
        .filter(|s| !s.trim().is_empty())
}

fn clean_model_output(text: &str) -> String {
    let mut result = text.trim().to_string();

    if let Some(pos) = result.rfind("\"translation\"") {
        result = result[pos..].to_string();
    }

    if let Some(pos) = result.rfind("Terjemahan:") {
        result = result[pos..].to_string();
    }

    result
        .replace("Thinking Process:", "")
        .replace("Analyze the Request:", "")
        .replace("Analyze the Context:", "")
        .trim()
        .to_string()
}

fn strip_think_blocks(text: &str) -> String {
    let mut out = text.to_string();
    while let (Some(start), Some(end)) = (out.find("<think>"), out.find("</think>")) {
        if end > start {
            let end_idx = end + "</think>".len();
            out.replace_range(start..end_idx, "");
        } else {
            break;
        }
    }
    out
}

fn sanitize_plain_translation_output(raw: &str) -> String {
    let mut s = strip_think_blocks(raw).trim().to_string();

    if let Some(stripped) = s.strip_prefix("```json") {
        s = stripped.trim().to_string();
    } else if let Some(stripped) = s.strip_prefix("```") {
        s = stripped.trim().to_string();
    }
    if let Some(stripped) = s.strip_suffix("```") {
        s = stripped.trim().to_string();
    }

    for prefix in [
        "Translation:",
        "Terjemahan:",
        "Final Answer:",
        "Jawaban:",
        "Answer:",
    ] {
        if let Some(stripped) = s.strip_prefix(prefix) {
            s = stripped.trim().to_string();
            break;
        }
    }

    s
}

fn looks_like_prompt_echo(text: &str) -> bool {
    let lowered = text.to_lowercase();

    lowered.contains("thinking process")
        || lowered.contains("analyze the request")
        || lowered.contains("analyze the context")
        || lowered.contains("input text")
        || lowered.contains("selected text")
        || lowered.contains("specific text to translate")
        || lowered.contains("context before")
        || lowered.contains("context after")
        || lowered.contains("constraint")
        || lowered.contains("task:")
        || lowered.contains("output format")
        || lowered.contains("critical rules")
        || lowered.contains("\"translation\": \"...\"")
}

fn is_translation_usable(translation: &str) -> bool {
    let trimmed = translation.trim();
    if trimmed.is_empty() {
        return false;
    }
    if looks_like_prompt_echo(trimmed) {
        return false;
    }
    true
}

// ============================================================================
// Default Prompts (hardcoded in backend)
// ============================================================================

// System instruction for translation (behavioral guidelines)
const TRANSLATION_SYSTEM_INSTRUCTION: &str = r#"You are a professional English-to-Indonesia translator and language teacher.

## Your Task
Translate ONLY the "SELECTED TEXT" provided by the user. The context is for understanding only.
IMPORTANT:
- DO NOT show thinking process.
- DO NOT explain your reasoing.
- DO NOT repeat the context.
- DO NOT write "Thinking Process".
- DO NOT write "Analyze the Request".
- ONLY return valid JSON.
"#;

// User prompt for translation (actual content - data only)
const TRANSLATION_PROMPT: &str = r#"Selected text:
{text}

Context:
{context_before} >>> {text} <<< {context_after}

Final Indonesian translation only:"#;

// System instruction for explanation (behavioral guidelines)
const EXPLANATION_SYSTEM_INSTRUCTION: &str = r#"You are an expert at explaining complex concepts in simple, easy-to-understand terms.

## Output Format (STRICT - follow exactly):
- Output MUST be valid JSON only. No markdown code blocks, no extra text.
- The JSON structure MUST be:
{
  "summary": "One-sentence summary (string)",
  "points": ["Point 1 (string)", "Point 2 (string)", "Point 3 (string)"]
}

## Critical Rules:
- The "points" field MUST be a flat array of strings. DO NOT use nested objects.
- All output text MUST be in Indonesian.

## Explanation Guidelines:

### Summary (summary field):
- Summarize the essence in ONE sentence
- Use phrases like "Intinya....." or "Artinya ......"
- Make it understandable even for someone unfamiliar with the topic

### Explanation points (points field):
- Rephrase technical terms in plain language: "X（artinya Y）"
- Use familiar analogies or metaphors to explain abstract concepts
- Add context about "why this matters" or "what benefit does this provide"
- For technical content, explain practical use cases and benefits concretely
- For academic content, explain the importance in the field and application examples
- Each point should be independently understandable
- Keep each point to 2-3 sentences"#;

// User prompt for explanation (actual content)
const EXPLANATION_PROMPT: &str = r#"Explain the following text.

The user has selected text from a PDF document. The context shows the surrounding text:
- "Context before" = text that appears BEFORE the selected text in the document
- "Text to explain" = the actual text the user selected
- "Context after" = text that appears AFTER the selected text in the document

## Context before (for understanding only):
{context_before}

## Text to explain:
{text}

## Context after (for understanding only):
{context_after}

Use the context to understand the meaning, but explain only the selected text."#;

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: u32,
    response_format: ResponseFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Debug, Serialize)]
struct ChatTemplateKwargs {
    enable_thinking: bool,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Option<Vec<ChatChoice>>,
    error: Option<ChatApiError>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    #[serde(default)]
    message: Option<ChatMessageResponse>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    #[serde(default)]
    content: Value,
    #[serde(default)]
    reasoning_content: Value,
    #[serde(default)]
    reasoning: Value,
}

#[derive(Debug, Deserialize)]
struct ChatApiError {
    message: String,
}

// ============================================================================
// Public Types
// ============================================================================

/// Structured translation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResponse {
    pub translation: String,
    pub points: Vec<String>,
}

/// Structured explanation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationResponse {
    pub summary: String,
    pub points: Vec<String>,
}

// ============================================================================
// API Functions
// ============================================================================

/// Call Gemini API with the given prompt and optional system instruction
async fn call_gemini_api(
    api_key: &str,
    model: &str,
    prompt: &str,
    system_instruction: Option<&str>,
) -> Result<String, PedaruError> {
    let timeout_seconds =
        env_u64("PEDARU_LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS, 30, 600);
    let max_tokens = env_u32("PEDARU_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS, 128, 2000);

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_seconds))
        .build()
        .map_err(|e| {
            PedaruError::Gemini(GeminiError::ApiRequestFailed(format!(
                "Failed to create HTTP client: {}",
                e
            )))
        })?;

    let mut messages = Vec::new();
    if let Some(instruction) = system_instruction {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: instruction.to_string(),
        });
    }
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: prompt.to_string(),
    });

    let request = ChatCompletionRequest {
        model: model.to_string(),
        messages,
        temperature: 0.1,
        max_tokens,
        response_format: ResponseFormat {
            format_type: "text".to_string(),
        },
        chat_template_kwargs: Some(ChatTemplateKwargs {
            enable_thinking: false,
        }),
    };

    let base_url = env::var("LM_STUDIO_BASE_URL")
        .or_else(|_| env::var("LLM_BASE_URL"))
        .unwrap_or_else(|_| DEFAULT_LM_STUDIO_API_BASE.to_string());
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

    let mut req = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&request);
    if !api_key.trim().is_empty() {
        req = req.bearer_auth(api_key.trim());
    }

    let response = req.send().await.map_err(|e| {
        let err_msg = if e.is_timeout() {
            format!(
                "Request timed out after {}s. Try a smaller/faster model or increase PEDARU_LLM_TIMEOUT_SECONDS.",
                timeout_seconds
            )
        } else if e.is_connect() {
            "Failed to connect to LM Studio local server. Make sure LM Studio is running at http://127.0.0.1:1234."
                .to_string()
        } else {
            format!("Network error: {}", e.without_url())
        };
        PedaruError::Gemini(GeminiError::ApiRequestFailed(err_msg))
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();

        let error_message = if status.as_u16() == 429 {
            "Local model is busy (rate limited). Please wait a moment and try again.".to_string()
        } else if status.as_u16() == 401 || status.as_u16() == 403 {
            "Invalid token/API key. Check LM Studio token in Settings (or leave empty if not required).".to_string()
        } else {
            format!("API error ({}): {}", status, error_text)
        };

        return Err(PedaruError::Gemini(GeminiError::ApiRequestFailed(
            error_message,
        )));
    }

    let completion_response: ChatCompletionResponse = response
        .json()
        .await
        .map_err(|e| PedaruError::Gemini(GeminiError::InvalidResponse(e.to_string())))?;

    if let Some(error) = completion_response.error.as_ref() {
        return Err(PedaruError::Gemini(GeminiError::ApiRequestFailed(
            error.message.clone(),
        )));
    }

    let text = completion_response
        .choices
        .and_then(|choices| choices.into_iter().next())
        .and_then(|choice| extract_chat_choice_text(&choice))
        .ok_or_else(|| {
            PedaruError::Gemini(GeminiError::InvalidResponse(
                "No text in response".to_string(),
            ))
        })?;

    Ok(text)
}

/// Parse JSON response from Gemini, with fallback for markdown code blocks
fn parse_translation_response(text: &str) -> Result<TranslationResponse, PedaruError> {
    eprintln!("[Gemini] Raw API response: {}", text);

    // Try to parse directly first
    if let Ok(response) = serde_json::from_str::<TranslationResponse>(text) {
        eprintln!("[Gemini] Parsed directly: {:?}", response);
        if !response.translation.trim().is_empty() || !response.points.is_empty() {
            return Ok(response);
        }
        eprintln!("[Gemini] Direct parse is empty, trying fallback parse...");
    }

    // Try to extract JSON from markdown code block
    let cleaned = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    eprintln!("[Gemini] Cleaned text: {}", cleaned);

    if let Ok(response) = serde_json::from_str::<TranslationResponse>(cleaned) {
        eprintln!("[Gemini] Parsed from cleaned: {:?}", response);
        if !response.translation.trim().is_empty() || !response.points.is_empty() {
            return Ok(response);
        }
        eprintln!("[Gemini] Cleaned parse is empty, trying flexible parse...");
    }

    // Try to parse as a more flexible JSON structure
    if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
        eprintln!("[Gemini] Parsed as Value: {:?}", value);

        // Handle both object and array responses
        let obj = if value.is_array() {
            // If it's an array, take the first element
            value.as_array().and_then(|arr| arr.first()).cloned()
        } else {
            Some(value)
        };

        if let Some(obj) = obj {
            let translation = value_to_text(&obj).unwrap_or_default();
            let points = extract_points(&obj);

            let response = if translation.trim().is_empty() && points.is_empty() {
                TranslationResponse {
                    translation: cleaned.to_string(),
                    points: vec![],
                }
            } else {
                TranslationResponse {
                    translation,
                    points,
                }
            };
            eprintln!("[Gemini] Flexible parse result: {:?}", response);
            return Ok(response);
        }
    }

    eprintln!("[Gemini] All parsing failed, returning raw text");
    // If all parsing fails, return the raw text as translation
    Ok(TranslationResponse {
        translation: text.to_string(),
        points: vec![],
    })
}

/// Parse JSON response for explanation, with fallback for markdown code blocks
fn parse_explanation_response(text: &str) -> Result<ExplanationResponse, PedaruError> {
    eprintln!("[Gemini] Raw API response (explanation): {}", text);

    // Try to parse directly first
    if let Ok(response) = serde_json::from_str::<ExplanationResponse>(text) {
        eprintln!("[Gemini] Parsed directly: {:?}", response);
        if !response.summary.trim().is_empty() || !response.points.is_empty() {
            return Ok(response);
        }
        eprintln!("[Gemini] Direct explanation parse is empty, trying fallback parse...");
    }

    // Try to extract JSON from markdown code block
    let cleaned = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    eprintln!("[Gemini] Cleaned text: {}", cleaned);

    if let Ok(response) = serde_json::from_str::<ExplanationResponse>(cleaned) {
        eprintln!("[Gemini] Parsed from cleaned: {:?}", response);
        if !response.summary.trim().is_empty() || !response.points.is_empty() {
            return Ok(response);
        }
        eprintln!("[Gemini] Cleaned explanation parse is empty, trying flexible parse...");
    }

    // Try to parse as a more flexible JSON structure
    if let Ok(value) = serde_json::from_str::<Value>(cleaned) {
        eprintln!("[Gemini] Parsed as Value: {:?}", value);

        // Handle both object and array responses
        let obj = if value.is_array() {
            value.as_array().and_then(|arr| arr.first()).cloned()
        } else {
            Some(value)
        };

        if let Some(obj) = obj {
            let summary = value_to_text(&obj).unwrap_or_default();
            let points = extract_points(&obj);

            let response = if summary.trim().is_empty() && points.is_empty() {
                ExplanationResponse {
                    summary: cleaned.to_string(),
                    points: vec![],
                }
            } else {
                ExplanationResponse { summary, points }
            };
            eprintln!("[Gemini] Flexible parse result: {:?}", response);
            return Ok(response);
        }
    }

    eprintln!("[Gemini] All parsing failed, returning raw text");
    Ok(ExplanationResponse {
        summary: text.to_string(),
        points: vec![],
    })
}

/// Translate text using Gemini API
///
/// Returns a structured response with translation and explanation points.
pub async fn translate_text(
    api_key: &str,
    model: &str,
    text: &str,
    context_before: &str,
    context_after: &str,
) -> Result<TranslationResponse, PedaruError> {
    let before = clamp_context(context_before, MAX_CONTEXT_CHARS);
    let after = clamp_context(context_after, MAX_CONTEXT_CHARS);
    let structured_prompt = TRANSLATION_PROMPT
        .replace("{text}", text)
        .replace("{context_before}", &before)
        .replace("{context_after}", &after);

    let response_text = call_gemini_api(
        api_key,
        model,
        &structured_prompt,
        Some(TRANSLATION_SYSTEM_INSTRUCTION),
    )
    .await?;
    let cleaned = sanitize_plain_translation_output(&response_text);

        if is_translation_usable(&cleaned) {
        return Ok(TranslationResponse {
            translation: cleaned,
            points: vec![],
        });
    }

    eprintln!("[Gemini] Structured mode returned unusable output, retrying with plain translation mode...");
    let plain_prompt = format!(
        "/no_think Terjemahkan teks bahasa Inggris berikut ke bahasa Indonesia.\n\
         Jawab HANYA dengan hasil terjemahan akhir tanpa penjelasan, tanpa daftar, tanpa JSON.\n\n\
         Teks: \"{}\"\n\
         Konteks sebelum: \"{}\"\n\
         Konteks sesudah: \"{}\"",
        text, before, after
    );

    let plain_response = call_gemini_api(api_key, model, &plain_prompt, None).await?;
    let cleaned_plain = sanitize_plain_translation_output(&plain_response);

    if is_translation_usable(&cleaned_plain) {
        return Ok(TranslationResponse {
            translation: cleaned_plain,
            points: vec![],
        });
    }

    eprintln!("[Gemini] Plain translation mode still unusable, returning best-effort cleaned text.");
    Ok(TranslationResponse {
        translation: if cleaned_plain.trim().is_empty() {
        "Model belum menghasilkan terjemahan yang bersih. Coba pilih teks lebih pendek.".to_string()
        } else {
            cleaned_plain
        },
        points: vec![],
    })
}

/// Get explanation of text
///
/// Returns a summary and explanation points.
/// The context parameters help understand the text but are not included in output.
pub async fn explain_text(
    api_key: &str,
    model: &str,
    text: &str,
    context_before: &str,
    context_after: &str,
) -> Result<ExplanationResponse, PedaruError> {
    let before = clamp_context(context_before, MAX_CONTEXT_CHARS);
    let after = clamp_context(context_after, MAX_CONTEXT_CHARS);
    let prompt = EXPLANATION_PROMPT
        .replace("{text}", text)
        .replace("{context_before}", &before)
        .replace("{context_after}", &after);

    let response_text = call_gemini_api(
        api_key,
        model,
        &prompt,
        Some(EXPLANATION_SYSTEM_INSTRUCTION),
    )
    .await?;
    parse_explanation_response(&response_text)
}
