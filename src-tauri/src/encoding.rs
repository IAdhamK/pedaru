//! PDF string encoding/decoding utilities
//!
//! This module handles multi-encoding support for PDF metadata strings.
//! Supports UTF-8, UTF-16BE, Shift-JIS, EUC-JP, ISO-2022-JP, and Latin-1.

use encoding_rs::SHIFT_JIS;

// ============================================================================
// Constants
// ============================================================================

/// List of Japanese encodings to try when decoding PDF strings
const JAPANESE_ENCODINGS: &[(&encoding_rs::Encoding, &str)] = &[
    (SHIFT_JIS, "Shift-JIS"),
    (encoding_rs::EUC_JP, "EUC-JP"),
    (encoding_rs::ISO_2022_JP, "ISO-2022-JP"),
];

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if bytes represent a UTF-16BE encoded string (with BOM)
///
/// UTF-16BE strings in PDF start with the byte order mark (BOM) 0xFE 0xFF.
pub fn is_utf16be(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF
}

/// Decode UTF-16BE bytes to a String
///
/// Expects bytes with or without BOM. If BOM is present, it is skipped.
/// Handles odd-length byte arrays gracefully by ignoring the trailing byte.
pub fn decode_utf16be(bytes: &[u8]) -> Option<String> {
    // Skip BOM if present
    let data = if is_utf16be(bytes) {
        &bytes[2..]
    } else {
        bytes
    };

    let utf16: Vec<u16> = data
        .chunks(2)
        .filter_map(|chunk| {
            if chunk.len() == 2 {
                Some(u16::from_be_bytes([chunk[0], chunk[1]]))
            } else {
                None
            }
        })
        .collect();

    String::from_utf16(&utf16).ok()
}

/// Decode bytes that may be UTF-16BE with BOM, or UTF-8/ASCII otherwise
///
/// This is a convenience function for decoding PDF string bytes that
/// could be either UTF-16BE (with BOM) or plain UTF-8/ASCII.
pub fn decode_utf16be_or_utf8(bytes: &[u8]) -> Option<String> {
    if is_utf16be(bytes) {
        decode_utf16be(bytes)
    } else {
        Some(String::from_utf8_lossy(bytes).to_string())
    }
}

/// Check if a character is a valid Japanese character
///
/// Returns true for hiragana, katakana, and common CJK ideographs.
#[inline]
fn is_japanese_char(c: char) -> bool {
    ('\u{3040}'..='\u{309F}').contains(&c) ||  // Hiragana
    ('\u{30A0}'..='\u{30FF}').contains(&c) ||  // Katakana
    ('\u{4E00}'..='\u{9FFF}').contains(&c) ||  // CJK Unified Ideographs
    ('\u{3400}'..='\u{4DBF}').contains(&c) // CJK Extension A
}

/// Decode bytes using Japanese encodings with scoring algorithm
///
/// Tries multiple Japanese encodings and selects the best result based on:
/// - Number of valid Japanese characters (positive score)
/// - Number of replacement characters (negative score)
/// - Number of control characters (negative score)
///
/// Returns the best decoded string, or None if no encoding worked.
fn try_japanese_encodings_with_scoring(bytes: &[u8], debug: bool) -> Option<String> {
    let mut best_result: Option<String> = None;
    let mut best_score = 0i32;

    for (encoding, name) in JAPANESE_ENCODINGS {
        let (decoded, _, had_errors) = encoding.decode(bytes);
        let decoded_str = decoded.into_owned();

        // Score the result: penalize replacement characters and control characters
        let replacement_count = decoded_str.chars().filter(|&c| c == '\u{FFFD}').count();
        let control_count = decoded_str
            .chars()
            .filter(|&c| c.is_control() && c != '\n' && c != '\r' && c != '\t')
            .count();
        let valid_japanese = decoded_str.chars().filter(|&c| is_japanese_char(c)).count() as i32;

        let score =
            valid_japanese * 10 - (replacement_count as i32 * 100) - (control_count as i32 * 50);

        if debug {
            eprintln!(
                "[Pedaru] Trying {}: had_errors={}, replacement={}, control={}, japanese={}, score={}, result={:?}",
                name,
                had_errors,
                replacement_count,
                control_count,
                valid_japanese,
                score,
                decoded_str
            );
        }

        if !had_errors && replacement_count == 0 && (best_result.is_none() || score > best_score) {
            best_result = Some(decoded_str);
            best_score = score;
        }
    }

    best_result
}

/// Try Japanese encodings and return first successful decode
///
/// Simpler version without scoring - returns the first encoding that
/// decodes without errors or replacement characters.
fn try_japanese_encodings_simple(bytes: &[u8]) -> Option<String> {
    for (encoding, _name) in JAPANESE_ENCODINGS {
        let (decoded, _, had_errors) = encoding.decode(bytes);
        let decoded_str = decoded.into_owned();
        let replacement_count = decoded_str.chars().filter(|&c| c == '\u{FFFD}').count();
        if !had_errors && replacement_count == 0 {
            return Some(decoded_str);
        }
    }
    None
}

/// Decode bytes to Latin-1/PDFDocEncoding
///
/// This is the fallback encoding when no other encoding works.
#[inline]
fn decode_latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| b as char).collect()
}

// ============================================================================
// Main Decoding Functions
// ============================================================================

/// Decode a PDF string object to a Rust String
///
/// PDF strings can be encoded in various formats:
/// - UTF-16BE (with BOM 0xFE 0xFF)
/// - UTF-8
/// - Japanese encodings (Shift-JIS, EUC-JP, ISO-2022-JP)
/// - Latin-1/PDFDocEncoding (fallback)
///
/// This function uses a scoring algorithm to select the best encoding
/// for Japanese text, prioritizing encodings that produce more valid
/// Japanese characters with fewer replacement characters.
pub fn decode_pdf_string(obj: &lopdf::Object) -> Option<String> {
    match obj {
        lopdf::Object::String(bytes, _) => {
            eprintln!(
                "[Pedaru] decode_pdf_string: bytes len={}, first bytes={:?}",
                bytes.len(),
                &bytes[..std::cmp::min(20, bytes.len())]
            );

            // Try UTF-16BE first (starts with BOM 0xFE 0xFF)
            if is_utf16be(bytes) {
                eprintln!("[Pedaru] Detected UTF-16BE");
                let result = decode_utf16be(bytes);
                eprintln!("[Pedaru] UTF-16BE result: {:?}", result);
                return result;
            }

            // Try UTF-8
            if let Ok(s) = String::from_utf8(bytes.clone()) {
                eprintln!("[Pedaru] Detected UTF-8: {:?}", s);
                return Some(s);
            }

            // Try Japanese encodings with scoring
            if let Some(result) = try_japanese_encodings_with_scoring(bytes, true) {
                eprintln!("[Pedaru] Best encoding result: {:?}", result);
                return Some(result);
            }

            // Fall back to Latin-1/PDFDocEncoding
            let result = decode_latin1(bytes);
            eprintln!("[Pedaru] Fallback to Latin-1: {:?}", result);
            Some(result)
        }
        _ => None,
    }
}

/// Decode a PDF name string object to a Rust String
///
/// Similar to `decode_pdf_string` but also handles `lopdf::Object::Name`
/// which is used for destination names in PDF outlines.
pub fn decode_name_string(obj: &lopdf::Object) -> Option<String> {
    match obj {
        lopdf::Object::String(bytes, _) => {
            // Try UTF-16BE first
            if is_utf16be(bytes) {
                return decode_utf16be(bytes);
            }

            // Try UTF-8
            if let Ok(s) = String::from_utf8(bytes.clone()) {
                return Some(s);
            }

            // Try Japanese encodings (simple, without scoring)
            if let Some(result) = try_japanese_encodings_simple(bytes) {
                return Some(result);
            }

            // Fallback to Latin-1
            Some(decode_latin1(bytes))
        }
        lopdf::Object::Name(bytes) => Some(String::from_utf8_lossy(bytes).to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_utf8_string() {
        // Test UTF-8 encoded string
        let utf8_bytes = "Hello World".as_bytes().to_vec();
        let obj = lopdf::Object::String(utf8_bytes, lopdf::StringFormat::Literal);
        let result = decode_pdf_string(&obj);
        assert_eq!(result, Some("Hello World".to_string()));
    }

    #[test]
    fn test_decode_utf16be_string() {
        // Test UTF-16BE with BOM (0xFE 0xFF)
        // "Hello" in UTF-16BE: FE FF 00 48 00 65 00 6C 00 6C 00 6F
        let mut utf16_bytes = vec![0xFE, 0xFF];
        utf16_bytes.extend_from_slice(&[0x00, 0x48]); // H
        utf16_bytes.extend_from_slice(&[0x00, 0x65]); // e
        utf16_bytes.extend_from_slice(&[0x00, 0x6C]); // l
        utf16_bytes.extend_from_slice(&[0x00, 0x6C]); // l
        utf16_bytes.extend_from_slice(&[0x00, 0x6F]); // o

        let obj = lopdf::Object::String(utf16_bytes, lopdf::StringFormat::Literal);
        let result = decode_pdf_string(&obj);
        assert_eq!(result, Some("Hello".to_string()));
    }

    #[test]
    fn test_decode_utf16be_multibyte_latin() {
        // Test multibyte Latin text "Café" in UTF-16BE
        // C: U+0043 (0x00, 0x43)
        // a: U+0061 (0x00, 0x61)
        // f: U+0066 (0x00, 0x66)
        // é: U+00E9 (0x00, 0xE9)
        let mut utf16_bytes = vec![0xFE, 0xFF]; // BOM
        utf16_bytes.extend_from_slice(&[0x00, 0x43]); // C
        utf16_bytes.extend_from_slice(&[0x00, 0x61]); // a
        utf16_bytes.extend_from_slice(&[0x00, 0x66]); // f
        utf16_bytes.extend_from_slice(&[0x00, 0xE9]); // é

        let obj = lopdf::Object::String(utf16_bytes, lopdf::StringFormat::Literal);
        let result = decode_pdf_string(&obj);
        assert_eq!(result, Some("Café".to_string()));
    }

    #[test]
    fn test_decode_shift_jis_compatible_ascii() {
        // ASCII bytes are compatible with Shift-JIS and should decode correctly.
        let shift_jis_bytes = b"ShiftJIS-Compatible".to_vec();
        let obj = lopdf::Object::String(shift_jis_bytes, lopdf::StringFormat::Literal);
        let result = decode_pdf_string(&obj);
        assert_eq!(result, Some("ShiftJIS-Compatible".to_string()));
    }

    #[test]
    fn test_decode_name_string_utf8() {
        // Test name string with UTF-8
        let utf8_bytes = "TestName".as_bytes().to_vec();
        let obj = lopdf::Object::String(utf8_bytes, lopdf::StringFormat::Literal);
        let result = decode_name_string(&obj);
        assert_eq!(result, Some("TestName".to_string()));
    }

    #[test]
    fn test_decode_name_string_from_name_object() {
        // Test decoding from lopdf::Object::Name
        let name_bytes = b"SomeName".to_vec();
        let obj = lopdf::Object::Name(name_bytes);
        let result = decode_name_string(&obj);
        assert_eq!(result, Some("SomeName".to_string()));
    }

    #[test]
    fn test_decode_empty_string() {
        // Test empty string
        let empty_bytes = vec![];
        let obj = lopdf::Object::String(empty_bytes, lopdf::StringFormat::Literal);
        let result = decode_pdf_string(&obj);
        // Empty string should return Some("")
        assert!(result.is_some());
    }

    #[test]
    fn test_decode_latin1_fallback() {
        // Test Latin-1/PDFDocEncoding fallback
        // Characters that are not valid UTF-8 but valid Latin-1
        let latin1_bytes = vec![0xE9, 0xE8, 0xE0]; // é è à in Latin-1
        let obj = lopdf::Object::String(latin1_bytes, lopdf::StringFormat::Literal);
        let result = decode_pdf_string(&obj);
        assert!(result.is_some());
        // Latin-1 characters should be decoded
        let decoded = result.unwrap();
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_decode_non_string_object() {
        // Test that non-string objects return None
        let obj = lopdf::Object::Integer(42);
        let result = decode_pdf_string(&obj);
        assert_eq!(result, None);
    }

    #[test]
    fn test_decode_name_string_non_string_object() {
        // Test that non-string/non-name objects return None
        let obj = lopdf::Object::Integer(42);
        let result = decode_name_string(&obj);
        assert_eq!(result, None);
    }

    #[test]
    fn test_decode_utf16be_with_odd_bytes() {
        // Test UTF-16BE with odd number of bytes (incomplete character)
        let mut utf16_bytes = vec![0xFE, 0xFF];
        utf16_bytes.extend_from_slice(&[0x00, 0x48]); // H
        utf16_bytes.push(0x00); // Incomplete character

        let obj = lopdf::Object::String(utf16_bytes, lopdf::StringFormat::Literal);
        let result = decode_pdf_string(&obj);
        // Should still decode what it can
        assert_eq!(result, Some("H".to_string()));
    }
}
