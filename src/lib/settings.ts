/**
 * Settings management functions
 *
 * This module provides functions to manage application settings,
 * particularly translation settings for local LM Studio.
 */

import { invoke } from "@tauri-apps/api/core";
import type {
  ExplanationResponse,
  GeminiModelOption,
  GeminiSettings,
  TranslationResponse,
} from "@/types";

// ============================================
// Default Values
// ============================================

export const DEFAULT_GEMINI_MODEL = "qwen/qwen3.5-9b";
export const DEFAULT_GEMINI_EXPLANATION_MODEL = "qwen/qwen3.5-9b";

export const DEFAULT_GEMINI_SETTINGS: GeminiSettings = {
  apiKey: "",
  model: DEFAULT_GEMINI_MODEL,
  explanationModel: DEFAULT_GEMINI_EXPLANATION_MODEL,
};

// ============================================
// Available Models
// ============================================

export const GEMINI_MODELS: GeminiModelOption[] = [
  {
    id: "qwen/qwen3.5-9b",
    name: "Qwen 3.5 9B",
    description: "LM Studio local model (Recommended)",
  },
  {
    id: "qwen/qwen3-8b",
    name: "Qwen 3 8B",
    description: "Fast local option",
  },
  {
    id: "meta-llama-3.1-8b-instruct",
    name: "Llama 3.1 8B Instruct",
    description: "Alternative local instruction model",
  },
];

// ============================================
// API Functions
// ============================================

/**
 * Get Gemini translation settings
 */
export async function getGeminiSettings(): Promise<GeminiSettings> {
  try {
    const settings = await invoke<GeminiSettings>("get_gemini_settings");
    return settings;
  } catch (error) {
    console.error("Failed to get Gemini settings:", error);
    return DEFAULT_GEMINI_SETTINGS;
  }
}

/**
 * Save Gemini translation settings
 */
export async function saveGeminiSettings(
  settings: GeminiSettings,
): Promise<void> {
  await invoke("save_gemini_settings", { settingsData: settings });
}

/**
 * Translate text using Gemini API
 * Returns a structured response with translation and points
 */
export async function translateWithGemini(
  text: string,
  contextBefore: string,
  contextAfter: string,
  modelOverride?: string,
): Promise<TranslationResponse> {
  const result = await invoke<TranslationResponse>("translate_with_gemini", {
    text,
    contextBefore,
    contextAfter,
    modelOverride: modelOverride ?? null,
  });
  return result;
}

/**
 * Get explanation of text
 * Returns summary + explanation points
 */
export async function explainDirectly(
  text: string,
  contextBefore: string,
  contextAfter: string,
  modelOverride?: string,
): Promise<ExplanationResponse> {
  const result = await invoke<ExplanationResponse>("explain_directly", {
    text,
    contextBefore,
    contextAfter,
    modelOverride: modelOverride ?? null,
  });
  return result;
}

/**
 * Check if translation is configured
 *
 * For LM Studio local usage, API key is optional.
 */
export async function isGeminiConfigured(): Promise<boolean> {
  const settings = await getGeminiSettings();
  return settings.model.trim().length > 0;
}
