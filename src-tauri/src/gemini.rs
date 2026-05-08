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
const AUTO_MODEL_ID: &str = "auto";
const DEFAULT_TIMEOUT_SECONDS: u64 = 90;
const DEFAULT_MAX_TOKENS: u32 = 700;
const MAX_TRANSLATION_CONTEXT_CHARS: usize = 500;
const MAX_EXPLANATION_CONTEXT_CHARS: usize = 800;

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

fn normalize_model_id(input: &str) -> &str {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        AUTO_MODEL_ID
    } else {
        trimmed
    }
}

fn is_supported_local_model_id(model_id: &str) -> bool {
    let lowered = model_id.to_lowercase();
    [
        "gemma",
        "llama",
        "mistral",
        "deepseek",
        "phi",
        "granite",
        "command-r",
        "gpt-oss",
    ]
    .iter()
    .any(|hint| lowered.contains(hint))
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
    strip_think_blocks(text)
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

    if let Ok(value) = serde_json::from_str::<Value>(&s) {
        if let Some(extracted) = value_to_text(&value) {
            s = extracted.trim().to_string();
        }
    }

    s
}

fn sanitize_explanation_output(raw: &str) -> String {
    strip_think_blocks(raw)
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim()
        .to_string()
}

fn looks_like_meta_label(text: &str) -> bool {
    let lowered = text.to_lowercase();
    lowered.starts_with("the user wants")
        || lowered.starts_with("analyze the input")
        || lowered.starts_with("analyze the request")
        || lowered.starts_with("context:")
        || lowered.starts_with("original word:")
        || lowered.starts_with("translation:")
        || lowered.starts_with("text to explain:")
        || lowered.starts_with("draft the output structure")
}

fn strip_list_prefix(line: &str) -> &str {
    let mut s = line.trim_start();
    for prefix in ["- ", "* ", "• "] {
        if let Some(rest) = s.strip_prefix(prefix) {
            return rest.trim();
        }
    }

    // Handle numbered points like "1. xxx" or "2) xxx".
    let mut idx = 0usize;
    let bytes = s.as_bytes();
    while idx < bytes.len() && bytes[idx].is_ascii_digit() {
        idx += 1;
    }
    if idx > 0 && idx < bytes.len() && (bytes[idx] == b'.' || bytes[idx] == b')') {
        idx += 1;
        while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        s = &s[idx..];
    }

    s.trim()
}

fn normalize_points(points: Vec<String>) -> Vec<String> {
    normalize_points_with_limit(points, 5)
}

fn normalize_points_with_limit(points: Vec<String>, limit: usize) -> Vec<String> {
    let mut normalized = Vec::new();

    for raw in points {
        let cleaned = sanitize_explanation_output(strip_list_prefix(&raw));
        if cleaned.is_empty() || looks_like_prompt_echo(&cleaned) || looks_like_meta_label(&cleaned)
        {
            continue;
        }
        if !normalized.iter().any(|p| p == &cleaned) {
            normalized.push(cleaned);
        }
    }

    normalized.into_iter().take(limit).collect()
}

fn parse_points_from_text(text: &str) -> Vec<String> {
    let cleaned = sanitize_explanation_output(text);
    if cleaned.is_empty() || looks_like_prompt_echo(&cleaned) {
        return vec![];
    }

    // Prefer JSON "points" when available.
    if let Ok(value) = serde_json::from_str::<Value>(&cleaned) {
        let points = normalize_points(extract_points(&value));
        if !points.is_empty() {
            return points;
        }
    }

    // Fallback: parse bullet/numbered lines.
    let mut lines = Vec::new();
    for line in cleaned.lines() {
        let point = strip_list_prefix(line).trim();
        if !point.is_empty() {
            lines.push(point.to_string());
        }
    }

    normalize_points(lines)
}

fn parse_points_from_json_only(text: &str) -> Vec<String> {
    let cleaned = sanitize_explanation_output(text);
    if cleaned.is_empty() || looks_like_prompt_echo(&cleaned) {
        return vec![];
    }

    if let Ok(value) = serde_json::from_str::<Value>(&cleaned) {
        return normalize_points_with_limit(extract_points(&value), 9);
    }

    vec![]
}

fn parse_explanation_from_plain_text(text: &str) -> Option<ExplanationResponse> {
    let cleaned = sanitize_explanation_output(text);
    if cleaned.is_empty() || looks_like_prompt_echo(&cleaned) {
        return None;
    }

    let mut lines: Vec<String> = cleaned
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();

    if lines.is_empty() {
        return None;
    }

    let mut summary = lines.remove(0);
    if let Some(rest) = summary.strip_prefix("Summary:") {
        summary = rest.trim().to_string();
    } else if let Some(rest) = summary.strip_prefix("Ringkasan:") {
        summary = rest.trim().to_string();
    }

    if summary.is_empty() || looks_like_prompt_echo(&summary) {
        return None;
    }

    let mut points = parse_points_from_text(&lines.join("\n"));
    if points.is_empty() && !lines.is_empty() {
        points = normalize_points(lines);
    }
    if points.is_empty() {
        points.push(format!("Makna ringkas: {}", summary));
    }

    Some(ExplanationResponse { summary, points })
}

fn parse_explanation_from_any_plain_text(text: &str) -> Option<ExplanationResponse> {
    let cleaned = sanitize_explanation_output(text);
    if cleaned.is_empty() || looks_like_prompt_echo(&cleaned) || looks_like_meta_label(&cleaned) {
        return None;
    }

    let sentence_split: Vec<String> = cleaned
        .split(['\n', '.', '!', '?'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    if sentence_split.is_empty() {
        return None;
    }

    let summary = sentence_split[0].clone();
    if summary.is_empty() || looks_like_prompt_echo(&summary) || looks_like_meta_label(&summary) {
        return None;
    }

    let mut points: Vec<String> = sentence_split
        .into_iter()
        .skip(1)
        .take(4)
        .filter(|line| !looks_like_prompt_echo(line) && !looks_like_meta_label(line))
        .collect();

    if points.is_empty() {
        points.push(format!("Makna ringkas: {}", summary));
    }

    Some(ExplanationResponse {
        summary: sanitize_explanation_output(&summary),
        points: normalize_points(points),
    })
}

fn tail_chars(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_string();
    }
    text.chars().skip(total - max_chars).collect()
}

fn head_chars(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

fn compact_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<&str>>().join(" ")
}

fn build_context_excerpt(context_before: &str, selected_text: &str, context_after: &str) -> String {
    let before = tail_chars(context_before, 120);
    let after = head_chars(context_after, 120);
    compact_whitespace(&format!("{} {} {}", before, selected_text, after))
}

fn highlight_first_occurrence(text: &str, needle: &str, replacement: &str) -> String {
    if needle.trim().is_empty() {
        return text.to_string();
    }
    if text.contains(needle) {
        text.replacen(needle, replacement, 1)
    } else {
        text.to_string()
    }
}

fn fallback_translation_points(
    selected_text: &str,
    translation: &str,
    context_before: &str,
    context_after: &str,
) -> Vec<String> {
    let context_excerpt = build_context_excerpt(context_before, selected_text, context_after);
    let has_context = !context_before.trim().is_empty() || !context_after.trim().is_empty();
    let word_count = selected_text.split_whitespace().count();
    let simple_type = if word_count > 1 {
        "phrase / frasa"
    } else {
        "kata; jenis pastinya mengikuti posisi di kalimat"
    };
    let context_basis = if has_context {
        "berdasarkan potongan kalimat di sekitar teks yang dipilih"
    } else {
        "berdasarkan teks yang dipilih"
    };
    let phrase = if has_context {
        highlight_first_occurrence(
            &context_excerpt,
            selected_text,
            &format!("***{}***", selected_text),
        )
    } else {
        "Tidak ada konteks kalimat yang tersedia.".to_string()
    };
    let translated_part = if has_context {
        highlight_first_occurrence(
            &context_excerpt,
            selected_text,
            &format!("***{}***", translation),
        )
    } else {
        translation.to_string()
    };

    vec![
        format!("Kata dipilih: {}", selected_text),
        format!("Jenis kata sederhana: {}", simple_type),
        format!("Arti inti: {}", translation),
        format!("Arti dalam konteks ini: {}.", context_basis),
        format!("Frasa lengkap: {}", phrase),
        format!("Terjemahan bagian kalimat: {}", translated_part),
        "Kata keluarga: belum tersedia dari model pada percobaan ini.".to_string(),
        format!(
            "Contoh pendek lain: \"{}\" dapat dipahami sebagai \"{}\" saat konteksnya cocok.",
            selected_text, translation
        ),
        format!(
            "Catatan: jangan memilih arti lain untuk \"{}\" jika tidak didukung oleh kata-kata di sekitarnya.",
            selected_text
        ),
    ]
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
        || lowered.contains("translate the provided")
        || lowered.contains("translate only")
        || lowered.contains("json structure")
        || lowered.contains("the specific text to translate")
        || lowered.contains("output only")
        || lowered.contains("do not show")
        || lowered.contains("professional english-to-indonesian translator")
        || lowered.contains("the user wants")
        || lowered.contains("analyze the input")
        || lowered.contains("original word")
        || lowered.contains("text to explain")
        || lowered.contains("determine the meaning")
        || lowered.contains("draft the output structure")
        || lowered.contains("must adhere")
        || lowered.contains("summary field")
        || lowered.contains("points field")
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

fn is_explanation_usable(explanation: &ExplanationResponse) -> bool {
    let summary = explanation.summary.trim();
    if summary.is_empty() || looks_like_prompt_echo(summary) {
        return false;
    }

    for point in &explanation.points {
        if looks_like_prompt_echo(point) {
            return false;
        }
    }

    true
}

// ============================================================================
// Default Prompts (hardcoded in backend)
// ============================================================================

// System instruction for translation (behavioral guidelines)
const TRANSLATION_SYSTEM_INSTRUCTION: &str = r#"You are an English-to-Indonesian translator for beginner learners.
Rules:
- Translate ONLY the selected text.
- Use the surrounding sentence/paragraph context to choose the right meaning.
- If the selected text has several meanings, choose the meaning that fits this context.
- Return only the final Indonesian translation in plain text.
- Do not return JSON, markdown, reasoning, task restatement, or context."#;

// User prompt for translation (actual content - data only)
const TRANSLATION_PROMPT: &str = r#"Selected text:
{text}

Context:
{context_before} >>> {text} <<< {context_after}

Return only the Indonesian translation:"#;

const TRANSLATION_POINTS_SYSTEM_INSTRUCTION: &str = r#"You generate concise Indonesian learning points for English learners.
Output valid JSON only:
{"points":["Kata dipilih: ...","Jenis kata sederhana: ...","Arti inti: ...","Arti dalam konteks ini: ...","Frasa lengkap: ...","Terjemahan bagian kalimat: ...","Kata keluarga: ...","Contoh pendek lain: ...","Catatan: ..."]}
Rules:
- All content must be in Indonesian except the selected English word/phrase and short English examples.
- Translation Points must focus on the selected word or phrase, not a broad paragraph explanation.
- Explain the meaning based on the surrounding sentence/paragraph context.
- If context is missing, say "berdasarkan teks yang dipilih".
- If the selected text is a phrase, explain it as one phrase.
- Mention only context-relevant meanings plus one short contrast if useful.
- Keep each point short and beginner-friendly.
- points must be plain strings (no nested objects).
- No reasoning text, no prompt restatement."#;

const TRANSLATION_POINTS_PROMPT: &str = r#"Buat Translation Points untuk membantu pemula memahami kata/frasa Inggris berikut.

Selected text:
{text}

Short Indonesian translation:
{translation}

Context before:
{context_before}

Context after:
{context_after}

Tugas:
- Fokus pada kata/frasa yang dipilih.
- Jelaskan arti inti dan arti dalam konteks kalimat ini.
- Ambil frasa lengkap dari konteks jika tersedia, misalnya "key terms" atau "sections form the extended content".
- Terjemahkan bagian kalimat yang paling relevan, bukan seluruh paragraf.
- Beri kata keluarga hanya jika berguna.
- Beri satu contoh pendek lain plus terjemahan Indonesia.
- Beri satu catatan pendek agar tidak salah menerjemahkan.
- Jangan memberi saran umum seperti "perhatikan kalimat sebelum dan sesudah" tanpa menjelaskan kalimat aktual.

Keluarkan JSON valid:
{"points":["Kata dipilih: {text}","Jenis kata sederhana: ...","Arti inti: ...","Arti dalam konteks ini: ...","Frasa lengkap: ...","Terjemahan bagian kalimat: ...","Kata keluarga: ...","Contoh pendek lain: ...","Catatan: ..."]}"#;

// System instruction for explanation (behavioral guidelines)
const EXPLANATION_SYSTEM_INSTRUCTION: &str = r#"You explain English text to Indonesian beginner learners.

## Output Format (STRICT - follow exactly):
- Output MUST be valid JSON only. No markdown code blocks, no extra text.
- The JSON structure MUST be:
{
  "summary": "Short Indonesian explanation (string)",
  "points": ["Point 1 (string)", "Point 2 (string)", "Point 3 (string)"]
}

## Critical Rules:
- The "points" field MUST be a flat array of strings. DO NOT use nested objects.
- All output text MUST be in Indonesian.
- Do not show hidden reasoning, chain-of-thought, prompt analysis, or raw JSON commentary.

## Explanation Guidelines:
- Explanation focuses on how the selected word/phrase works inside the surrounding sentence or paragraph.
- Even if the selected text is only one word, also explain the sentence part around that word.
- Explain why the selected word has that meaning in this context.
- Explain how it connects to nearby words.
- If enough paragraph context is available, summarize the relevant paragraph idea in simple Indonesian.
- If context is missing, say "berdasarkan teks yang dipilih" and avoid pretending you know the sentence.
- Avoid long grammar lectures. If a grammar term is needed, explain it immediately in simple Indonesian.
- Keep the summary and points direct, concise, and beginner-friendly."#;

// User prompt for explanation (actual content)
const EXPLANATION_PROMPT: &str = r#"Jelaskan selected text berikut untuk pembelajar bahasa Inggris.

Context before:
{context_before}

Selected text:
{text}

Context after:
{context_after}

Tugas Explanation:
- Fokus pada cara "{text}" bekerja di dalam kalimat/paragraf sekitar.
- Jangan hanya menerjemahkan "{text}" secara terpisah.
- Jelaskan mengapa artinya begitu dalam konteks ini.
- Jelaskan hubungan "{text}" dengan kata-kata terdekat.
- Jelaskan arti bagian kalimat yang memuat "{text}".
- Jika konteks paragraf cukup, ringkas ide paragraf yang relevan dalam bahasa Indonesia sederhana.
- Jika konteks tidak cukup, sebutkan "berdasarkan teks yang dipilih".

Keluarkan JSON valid saja:
{"summary":"...","points":["...","...","..."]}"#;

const EXPLANATION_FALLBACK_PLAIN_SYSTEM_INSTRUCTION: &str = r#"Jelaskan cara kata/frasa bekerja dalam konteks kalimat/paragraf memakai Bahasa Indonesia sederhana.
Format plain text:
Summary: ...
- ...
- ...
Tanpa reasoning, tanpa menyalin instruksi, dan jangan memberi saran umum tanpa menjelaskan teks aktual."#;

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

fn parse_model_ids(value: &Value) -> Vec<String> {
    let mut ids = Vec::new();
    if let Some(items) = value.get("data").and_then(|v| v.as_array()) {
        for item in items {
            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                let id_trimmed = id.trim();
                if !id_trimmed.is_empty() {
                    ids.push(id_trimmed.to_string());
                }
            }
        }
    }

    ids
}

fn extract_model_size_billion(id: &str) -> Option<f32> {
    let chars: Vec<char> = id.to_lowercase().chars().collect();
    let mut best: Option<f32> = None;

    for idx in 1..chars.len() {
        if chars[idx] != 'b' {
            continue;
        }
        let mut start = idx;
        while start > 0 && (chars[start - 1].is_ascii_digit() || chars[start - 1] == '.') {
            start -= 1;
        }
        if start < idx {
            let num: String = chars[start..idx].iter().collect();
            if let Ok(v) = num.parse::<f32>() {
                best = Some(best.map_or(v, |curr| curr.min(v)));
            }
        }
    }

    best
}

fn choose_auto_model_id(model_ids: &[String]) -> Option<String> {
    if model_ids.is_empty() {
        return None;
    }

    // Optional override via env var.
    if let Ok(preferred_raw) = env::var("PEDARU_LM_STUDIO_AUTO_MODEL") {
        let preferred = preferred_raw.trim();
        if !preferred.is_empty() && is_supported_local_model_id(preferred) {
            if let Some(existing) = model_ids
                .iter()
                .find(|id| id.eq_ignore_ascii_case(preferred) && is_supported_local_model_id(id))
            {
                return Some(existing.clone());
            }
            return Some(preferred.to_string());
        }
    }

    // Prefer the smallest supported loaded model by size hint in ID (e.g. 2b < 4b < 9b).
    // If size hint is unavailable, keep it as lower priority fallback.
    model_ids
        .iter()
        .filter(|id| is_supported_local_model_id(id))
        .cloned()
        .min_by(|a, b| {
            let sa = extract_model_size_billion(a).unwrap_or(9999.0);
            let sb = extract_model_size_billion(b).unwrap_or(9999.0);
            sa.partial_cmp(&sb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        })
}

async fn resolve_model_id(
    client: &Client,
    api_key: &str,
    base_url: &str,
    requested_model: &str,
) -> Result<String, PedaruError> {
    let normalized = normalize_model_id(requested_model);
    if !normalized.eq_ignore_ascii_case(AUTO_MODEL_ID) {
        if !is_supported_local_model_id(normalized) {
            return Err(PedaruError::Gemini(GeminiError::ApiRequestFailed(
                "Unsupported local model ID. Use a supported model family (Gemma, Llama, Mistral, DeepSeek, Phi, Granite, Command-R, GPT-OSS) or set model to auto."
                    .to_string(),
            )));
        }
        return Ok(normalized.to_string());
    }

    let models_url = format!("{}/models", base_url.trim_end_matches('/'));
    let mut req = client.get(&models_url);
    if !api_key.trim().is_empty() {
        req = req.bearer_auth(api_key.trim());
    }

    let response = req.send().await.map_err(|e| {
        PedaruError::Gemini(GeminiError::ApiRequestFailed(format!(
            "Failed to query LM Studio models endpoint: {}",
            e.without_url()
        )))
    })?;

    if !response.status().is_success() {
        return Err(PedaruError::Gemini(GeminiError::ApiRequestFailed(format!(
            "Failed to get model list from LM Studio ({}). Please ensure at least one model is loaded.",
            response.status()
        ))));
    }

    let payload: Value = response
        .json()
        .await
        .map_err(|e| PedaruError::Gemini(GeminiError::InvalidResponse(e.to_string())))?;

    let model_ids = parse_model_ids(&payload);
    choose_auto_model_id(&model_ids).ok_or_else(|| {
        PedaruError::Gemini(GeminiError::ApiRequestFailed(
            "LM Studio did not return a supported loaded model ID. Load a supported model (for example Gemma) or set model ID manually in Settings."
                .to_string(),
        ))
    })
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
    let timeout_seconds = env_u64(
        "PEDARU_LLM_TIMEOUT_SECONDS",
        DEFAULT_TIMEOUT_SECONDS,
        30,
        600,
    );
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

    let base_url = env::var("LM_STUDIO_BASE_URL")
        .or_else(|_| env::var("LLM_BASE_URL"))
        .unwrap_or_else(|_| DEFAULT_LM_STUDIO_API_BASE.to_string());
    let resolved_model = resolve_model_id(&client, api_key, &base_url, model).await?;

    let request = ChatCompletionRequest {
        model: resolved_model,
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
    if let Ok(value) = serde_json::from_str::<Value>(&cleaned) {
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
    if let Ok(mut response) = serde_json::from_str::<ExplanationResponse>(text) {
        response.summary = sanitize_explanation_output(&response.summary);
        response.points = response
            .points
            .into_iter()
            .map(|p| sanitize_explanation_output(&p))
            .filter(|p| !p.is_empty())
            .collect();
        eprintln!("[Gemini] Parsed directly: {:?}", response);
        if is_explanation_usable(&response) {
            return Ok(response);
        }
        eprintln!("[Gemini] Direct explanation parse is empty, trying fallback parse...");
    }

    // Try to extract JSON from markdown code block
    let cleaned = sanitize_explanation_output(text);

    eprintln!("[Gemini] Cleaned text: {}", cleaned);

    if let Ok(mut response) = serde_json::from_str::<ExplanationResponse>(&cleaned) {
        response.summary = sanitize_explanation_output(&response.summary);
        response.points = response
            .points
            .into_iter()
            .map(|p| sanitize_explanation_output(&p))
            .filter(|p| !p.is_empty())
            .collect();
        eprintln!("[Gemini] Parsed from cleaned: {:?}", response);
        if is_explanation_usable(&response) {
            return Ok(response);
        }
        eprintln!("[Gemini] Cleaned explanation parse is empty, trying flexible parse...");
    }

    // Try to parse as a more flexible JSON structure
    if let Ok(value) = serde_json::from_str::<Value>(&cleaned) {
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
                ExplanationResponse {
                    summary: sanitize_explanation_output(&summary),
                    points: points
                        .into_iter()
                        .map(|p| sanitize_explanation_output(&p))
                        .filter(|p| !p.is_empty())
                        .collect(),
                }
            };
            eprintln!("[Gemini] Flexible parse result: {:?}", response);
            if is_explanation_usable(&response) {
                return Ok(response);
            }
        }
    }

    Err(PedaruError::Gemini(GeminiError::InvalidResponse(
        "Explanation response unusable".to_string(),
    )))
}

async fn generate_translation_points(
    api_key: &str,
    model: &str,
    text: &str,
    translation: &str,
    context_before: &str,
    context_after: &str,
) -> Vec<String> {
    let prompt = TRANSLATION_POINTS_PROMPT
        .replace("{text}", text)
        .replace("{translation}", translation)
        .replace("{context_before}", context_before)
        .replace("{context_after}", context_after);

    match call_gemini_api(
        api_key,
        model,
        &prompt,
        Some(TRANSLATION_POINTS_SYSTEM_INSTRUCTION),
    )
    .await
    {
        Ok(raw) => {
            // Translation points must come from structured data only.
            // Plain-text fallback often contains prompt-echo on smaller local models.
            let points = parse_points_from_json_only(&raw);
            if !points.is_empty() {
                return points;
            }
            vec![]
        }
        Err(e) => {
            eprintln!("[Gemini] Failed to generate translation points: {}", e);
            vec![]
        }
    }
}

async fn build_translation_response(
    api_key: &str,
    model: &str,
    selected_text: &str,
    context_before: &str,
    context_after: &str,
    translation: String,
    points: Vec<String>,
) -> TranslationResponse {
    let normalized_existing = normalize_points_with_limit(points, 9);
    if !normalized_existing.is_empty() {
        return TranslationResponse {
            translation,
            points: normalized_existing,
        };
    }

    let generated_points = generate_translation_points(
        api_key,
        model,
        selected_text,
        &translation,
        context_before,
        context_after,
    )
    .await;

    let points = if generated_points.is_empty() {
        fallback_translation_points(selected_text, &translation, context_before, context_after)
    } else {
        generated_points
    };

    TranslationResponse {
        translation,
        points,
    }
}

/// Translate text using Gemini API
///
/// Returns a structured response with translation and word/phrase learning points.
pub async fn translate_text(
    api_key: &str,
    model: &str,
    text: &str,
    context_before: &str,
    context_after: &str,
) -> Result<TranslationResponse, PedaruError> {
    let before = tail_chars(context_before, MAX_TRANSLATION_CONTEXT_CHARS);
    let after = head_chars(context_after, MAX_TRANSLATION_CONTEXT_CHARS);
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

    if let Ok(parsed) = parse_translation_response(&response_text) {
        let parsed_translation = sanitize_plain_translation_output(&parsed.translation);
        if is_translation_usable(&parsed_translation) {
            return Ok(build_translation_response(
                api_key,
                model,
                text,
                &before,
                &after,
                parsed_translation,
                parsed.points,
            )
            .await);
        }
    }

    let cleaned = sanitize_plain_translation_output(&response_text);

    if is_translation_usable(&cleaned) {
        return Ok(build_translation_response(
            api_key,
            model,
            text,
            &before,
            &after,
            cleaned,
            vec![],
        )
        .await);
    }

    eprintln!(
        "[Gemini] Structured mode returned unusable output, retrying with plain translation mode..."
    );
    let plain_prompt = format!(
        "Translate this English text to Indonesian.\n\
         Use the context only to choose the right meaning.\n\
         Reply with the translation only (plain text, no JSON, no explanation, no context).\n\n\
         Text: {}\n\
         Context: {} >>> {} <<< {}",
        text, before, text, after
    );

    let plain_response = call_gemini_api(
        api_key,
        model,
        &plain_prompt,
        Some(
            "Return only the final Indonesian translation in one line. No thinking. No restating the task.",
        ),
    )
    .await?;
    let cleaned_plain = sanitize_plain_translation_output(&plain_response);

    if is_translation_usable(&cleaned_plain) {
        return Ok(build_translation_response(
            api_key,
            model,
            text,
            &before,
            &after,
            cleaned_plain,
            vec![],
        )
        .await);
    }

    eprintln!("[Gemini] Translation output is still unusable after retry.");
    Ok(TranslationResponse {
        translation: "Model belum menghasilkan terjemahan yang valid. Coba lagi dengan teks lebih pendek atau model LM Studio lain.".to_string(),
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
    let before = tail_chars(context_before, MAX_EXPLANATION_CONTEXT_CHARS);
    let after = head_chars(context_after, MAX_EXPLANATION_CONTEXT_CHARS);
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

    if let Ok(parsed) = parse_explanation_response(&response_text) {
        if is_explanation_usable(&parsed) {
            return Ok(parsed);
        }
    }
    if let Some(parsed) = parse_explanation_from_any_plain_text(&response_text) {
        if is_explanation_usable(&parsed) {
            return Ok(parsed);
        }
    }

    eprintln!("[Gemini] Structured explanation unusable, retrying with strict short prompt...");
    let fallback_prompt = format!(
        "Jelaskan selected text berikut dalam Bahasa Indonesia untuk pembelajar pemula.\n\
         Keluarkan JSON valid saja dengan format:\n\
         {{\"summary\":\"...\",\"points\":[\"...\",\"...\"]}}\n\
         Fokus pada arti selected text di dalam kalimat/paragraf sekitar.\n\
         Jika konteks kosong, sebutkan \"berdasarkan teks yang dipilih\".\n\
         Jangan tampilkan proses berpikir.\n\n\
         Selected text: {}\n\
         Context before: {}\n\
         Context after: {}",
        text, before, after
    );

    if let Ok(fallback_text) = call_gemini_api(
        api_key,
        model,
        &fallback_prompt,
        Some(
            "Jawab hanya JSON valid dengan field summary dan points. Tanpa reasoning, tanpa instruksi ulang.",
        ),
    )
    .await
    {
        if let Ok(parsed) = parse_explanation_response(&fallback_text) {
            if is_explanation_usable(&parsed) {
                return Ok(parsed);
            }
        }
        if let Some(parsed) = parse_explanation_from_any_plain_text(&fallback_text) {
            if is_explanation_usable(&parsed) {
                return Ok(parsed);
            }
        }
    } else {
        eprintln!("[Gemini] JSON explanation fallback request failed.");
    }

    eprintln!("[Gemini] JSON explanation fallback unusable, retrying with plain-text fallback...");
    let plain_fallback_prompt = format!(
        "Jelaskan selected text berikut dalam Bahasa Indonesia secara ringkas.\n\
         Jawab dengan format ini:\n\
         Summary: ...\n\
         - ...\n\
         - ...\n\
         Jelaskan arti kata/frasa ini di dalam bagian kalimat terdekat.\n\
         Jika konteks kosong, sebutkan \"berdasarkan teks yang dipilih\".\n\
         Jangan tampilkan proses berpikir.\n\n\
         Selected text: {}\n\
         Konteks sebelum: {}\n\
         Konteks sesudah: {}",
        text, before, after
    );
    if let Ok(plain_fallback_text) = call_gemini_api(
        api_key,
        model,
        &plain_fallback_prompt,
        Some(EXPLANATION_FALLBACK_PLAIN_SYSTEM_INSTRUCTION),
    )
    .await
    {
        if let Some(parsed) = parse_explanation_from_plain_text(&plain_fallback_text) {
            if is_explanation_usable(&parsed) {
                return Ok(parsed);
            }
        }
        if let Some(parsed) = parse_explanation_from_any_plain_text(&plain_fallback_text) {
            if is_explanation_usable(&parsed) {
                return Ok(parsed);
            }
        }
    } else {
        eprintln!("[Gemini] Plain explanation fallback request failed.");
    }

    let context_excerpt = build_context_excerpt(&before, text, &after);
    let has_context = !before.trim().is_empty() || !after.trim().is_empty();
    let summary = if has_context {
        format!(
            "Artinya, \"{}\" perlu dipahami dari bagian kalimat di sekitarnya, bukan dari arti kamus saja.",
            text
        )
    } else {
        format!(
            "Artinya, \"{}\" dijelaskan berdasarkan teks yang dipilih karena konteks kalimat tidak tersedia.",
            text
        )
    };

    let points = if has_context {
        vec![
            format!(
                "Bagian kalimat terdekat: {}",
                highlight_first_occurrence(&context_excerpt, text, &format!("***{}***", text))
            ),
            format!(
                "Hubungan kata: \"{}\" harus dibaca bersama kata-kata di dekatnya dalam potongan itu agar maknanya tidak bergeser.",
                text
            ),
        ]
    } else {
        vec![
            format!(
                "Karena konteks tidak tersedia, penjelasan ini hanya berdasarkan teks yang dipilih: \"{}\".",
                text
            ),
            "Pilih arti yang paling cocok setelah kalimat lengkap tersedia.".to_string(),
        ]
    };

    Ok(ExplanationResponse { summary, points })
}
