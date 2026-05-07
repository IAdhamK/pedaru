//! Integration tests for PDF processing
//!
//! These tests use dynamically generated PDF files and fixture files to test
//! the full PDF processing pipeline.
//!
//! ## Test Fixtures
//!
//! PDF fixture files are located in `tests/fixtures/`:
//! - `encrypted_empty_password.pdf` - Simple encrypted PDF with empty user password

use lopdf::{
    Document, EncryptionState, EncryptionVersion, Object, ObjectId, Permissions, Stream,
    StringFormat,
};
use pedaru_lib::pdf::extract_toc;
use pedaru_lib::types::TocEntry;
use std::io::Write;
use std::path::PathBuf;
use tempfile::NamedTempFile;

/// Get the path to a test fixture file
fn fixture_path(filename: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("fixtures");
    path.push(filename);
    path
}

/// Load an encrypted PDF fixture file
fn load_encrypted_pdf_fixture(filename: &str) -> Document {
    let path = fixture_path(filename);
    Document::load(&path).unwrap_or_else(|_| panic!("Failed to load fixture: {}", filename))
}

/// Create a simple PDF document with specified number of pages
fn create_simple_pdf(num_pages: u32) -> Document {
    let mut doc = Document::with_version("1.5");
    let mut pages_kids: Vec<Object> = Vec::new();
    let pages_id = doc.new_object_id();

    // Create page objects
    for i in 1..=num_pages {
        let content_id = doc.add_object(Stream::new(
            lopdf::Dictionary::new(),
            format!("BT /F1 12 Tf 100 700 Td (Page {}) Tj ET", i).into_bytes(),
        ));

        let mut page_dict = lopdf::Dictionary::new();
        page_dict.set("Type", Object::Name(b"Page".to_vec()));
        page_dict.set("Parent", Object::Reference(pages_id));
        page_dict.set(
            "MediaBox",
            Object::Array(vec![
                Object::Integer(0),
                Object::Integer(0),
                Object::Integer(612),
                Object::Integer(792),
            ]),
        );
        page_dict.set("Contents", Object::Reference(content_id));

        let page_id = doc.add_object(page_dict);
        pages_kids.push(Object::Reference(page_id));
    }

    // Create Pages dictionary
    let mut pages_dict = lopdf::Dictionary::new();
    pages_dict.set("Type", Object::Name(b"Pages".to_vec()));
    pages_dict.set("Kids", Object::Array(pages_kids));
    pages_dict.set("Count", Object::Integer(num_pages as i64));
    doc.objects.insert(pages_id, Object::Dictionary(pages_dict));

    // Create Catalog
    let mut catalog = lopdf::Dictionary::new();
    catalog.set("Type", Object::Name(b"Catalog".to_vec()));
    catalog.set("Pages", Object::Reference(pages_id));

    let catalog_id = doc.add_object(catalog);
    doc.trailer.set("Root", Object::Reference(catalog_id));

    doc
}

/// Create a PDF with TOC (outline) structure
fn create_pdf_with_toc() -> Document {
    let mut doc = create_simple_pdf(5);
    let pages = doc.get_pages();

    // Get catalog
    let catalog_id = match doc.trailer.get(b"Root") {
        Ok(Object::Reference(id)) => *id,
        _ => panic!("No catalog found"),
    };

    // Create outline items
    let page_ids: Vec<ObjectId> = pages.values().cloned().collect();

    // Create child outline item
    let child_outline_id = doc.new_object_id();
    let mut child_outline = lopdf::Dictionary::new();
    child_outline.set(
        "Title",
        Object::String(b"Section 1.1".to_vec(), StringFormat::Literal),
    );
    child_outline.set(
        "Dest",
        Object::Array(vec![
            Object::Reference(page_ids[1]), // Page 2
            Object::Name(b"Fit".to_vec()),
        ]),
    );

    // Create first outline item with child
    let first_outline_id = doc.new_object_id();
    let mut first_outline = lopdf::Dictionary::new();
    first_outline.set(
        "Title",
        Object::String(b"Chapter 1".to_vec(), StringFormat::Literal),
    );
    first_outline.set(
        "Dest",
        Object::Array(vec![
            Object::Reference(page_ids[0]), // Page 1
            Object::Name(b"Fit".to_vec()),
        ]),
    );
    first_outline.set("First", Object::Reference(child_outline_id));
    first_outline.set("Last", Object::Reference(child_outline_id));

    // Set parent for child
    child_outline.set("Parent", Object::Reference(first_outline_id));
    doc.objects
        .insert(child_outline_id, Object::Dictionary(child_outline));

    // Create second outline item
    let second_outline_id = doc.new_object_id();
    let mut second_outline = lopdf::Dictionary::new();
    second_outline.set(
        "Title",
        Object::String(b"Chapter 2".to_vec(), StringFormat::Literal),
    );
    second_outline.set(
        "Dest",
        Object::Array(vec![
            Object::Reference(page_ids[2]), // Page 3
            Object::Name(b"Fit".to_vec()),
        ]),
    );

    // Link outline items
    first_outline.set("Next", Object::Reference(second_outline_id));
    second_outline.set("Prev", Object::Reference(first_outline_id));

    // Create Outlines dictionary
    let outlines_id = doc.new_object_id();
    let mut outlines = lopdf::Dictionary::new();
    outlines.set("Type", Object::Name(b"Outlines".to_vec()));
    outlines.set("First", Object::Reference(first_outline_id));
    outlines.set("Last", Object::Reference(second_outline_id));
    outlines.set("Count", Object::Integer(3)); // 2 top-level + 1 child

    // Set parents for top-level items
    first_outline.set("Parent", Object::Reference(outlines_id));
    second_outline.set("Parent", Object::Reference(outlines_id));

    doc.objects
        .insert(first_outline_id, Object::Dictionary(first_outline));
    doc.objects
        .insert(second_outline_id, Object::Dictionary(second_outline));
    doc.objects
        .insert(outlines_id, Object::Dictionary(outlines));

    // Add Outlines to catalog
    if let Ok(Object::Dictionary(cat_dict)) = doc.get_object_mut(catalog_id) {
        cat_dict.set("Outlines", Object::Reference(outlines_id));
    }

    doc
}

/// Create a PDF with UTF-16BE metadata
fn create_pdf_with_utf16_metadata() -> Document {
    let mut doc = create_simple_pdf(1);

    // Create Info dictionary with UTF-16BE text
    let mut info = lopdf::Dictionary::new();

    // UTF-16BE encoded "Unicode Title" with BOM
    let title_bytes = create_utf16be_string("Unicode Title");
    info.set("Title", Object::String(title_bytes, StringFormat::Literal));

    // UTF-16BE encoded "Author Name" with BOM
    let author_bytes = create_utf16be_string("Author Name");
    info.set(
        "Author",
        Object::String(author_bytes, StringFormat::Literal),
    );

    let info_id = doc.add_object(info);
    doc.trailer.set("Info", Object::Reference(info_id));

    doc
}

/// Create a 3-page PDF with English metadata and TOC structure
fn create_pdf_with_english_metadata_and_toc() -> Document {
    let mut doc = create_simple_pdf(3);
    let pages = doc.get_pages();

    let catalog_id = match doc.trailer.get(b"Root") {
        Ok(Object::Reference(id)) => *id,
        _ => panic!("No catalog found"),
    };

    let page_ids: Vec<ObjectId> = pages.values().cloned().collect();

    let child_outline_id = doc.new_object_id();
    let mut child_outline = lopdf::Dictionary::new();
    child_outline.set(
        "Title",
        Object::String(
            b"Section 1.1 Overview".to_vec(),
            StringFormat::Literal,
        ),
    );
    child_outline.set(
        "Dest",
        Object::Array(vec![
            Object::Reference(page_ids[1]),
            Object::Name(b"Fit".to_vec()),
        ]),
    );

    let first_outline_id = doc.new_object_id();
    let mut first_outline = lopdf::Dictionary::new();
    first_outline.set(
        "Title",
        Object::String(
            b"Chapter 1 Introduction".to_vec(),
            StringFormat::Literal,
        ),
    );
    first_outline.set(
        "Dest",
        Object::Array(vec![
            Object::Reference(page_ids[0]),
            Object::Name(b"Fit".to_vec()),
        ]),
    );
    first_outline.set("First", Object::Reference(child_outline_id));
    first_outline.set("Last", Object::Reference(child_outline_id));

    child_outline.set("Parent", Object::Reference(first_outline_id));
    doc.objects
        .insert(child_outline_id, Object::Dictionary(child_outline));

    let second_outline_id = doc.new_object_id();
    let mut second_outline = lopdf::Dictionary::new();
    second_outline.set(
        "Title",
        Object::String(
            b"Chapter 2 Main Discussion".to_vec(),
            StringFormat::Literal,
        ),
    );
    second_outline.set(
        "Dest",
        Object::Array(vec![
            Object::Reference(page_ids[2]),
            Object::Name(b"Fit".to_vec()),
        ]),
    );

    first_outline.set("Next", Object::Reference(second_outline_id));
    second_outline.set("Prev", Object::Reference(first_outline_id));

    let outlines_id = doc.new_object_id();
    let mut outlines = lopdf::Dictionary::new();
    outlines.set("Type", Object::Name(b"Outlines".to_vec()));
    outlines.set("First", Object::Reference(first_outline_id));
    outlines.set("Last", Object::Reference(second_outline_id));
    outlines.set("Count", Object::Integer(3));

    first_outline.set("Parent", Object::Reference(outlines_id));
    second_outline.set("Parent", Object::Reference(outlines_id));

    doc.objects
        .insert(first_outline_id, Object::Dictionary(first_outline));
    doc.objects
        .insert(second_outline_id, Object::Dictionary(second_outline));
    doc.objects
        .insert(outlines_id, Object::Dictionary(outlines));

    if let Ok(Object::Dictionary(cat_dict)) = doc.get_object_mut(catalog_id) {
        cat_dict.set("Outlines", Object::Reference(outlines_id));
    }

    let mut info = lopdf::Dictionary::new();
    info.set(
        "Title",
        Object::String(
            create_utf16be_string("Encrypted Test Document"),
            StringFormat::Literal,
        ),
    );
    info.set(
        "Author",
        Object::String(
            create_utf16be_string("Sample Author"),
            StringFormat::Literal,
        ),
    );
    let info_id = doc.add_object(info);
    doc.trailer.set("Info", Object::Reference(info_id));

    doc
}

/// Create and load a generated encrypted English PDF for integration tests
fn load_generated_encrypted_english_pdf() -> Document {
    let mut doc = create_pdf_with_english_metadata_and_toc();
    let version = EncryptionVersion::V2 {
        document: &doc,
        owner_password: "owner",
        user_password: "",
        key_length: 40,
        permissions: Permissions::all(),
    };
    let state =
        EncryptionState::try_from(version).expect("Failed to build encryption state for test");
    doc.encrypt(&state)
        .expect("Failed to encrypt generated English test PDF");

    let temp_file = save_to_temp_file(&mut doc);
    Document::load(temp_file.path()).expect("Failed to load generated encrypted English test PDF")
}

/// Create UTF-16BE encoded string with BOM
fn create_utf16be_string(s: &str) -> Vec<u8> {
    let mut bytes = vec![0xFE, 0xFF]; // BOM
    for c in s.chars() {
        let code = c as u32;
        if code <= 0xFFFF {
            bytes.push((code >> 8) as u8);
            bytes.push((code & 0xFF) as u8);
        }
    }
    bytes
}

/// Save document to a temporary file
fn save_to_temp_file(doc: &mut Document) -> NamedTempFile {
    let mut temp_file = NamedTempFile::with_suffix(".pdf").expect("Failed to create temp file");
    doc.save_to(temp_file.as_file_mut())
        .expect("Failed to save PDF");
    temp_file.as_file_mut().flush().expect("Failed to flush");
    temp_file
}

#[test]
fn test_extract_toc_empty_pdf() {
    let doc = create_simple_pdf(3);
    let toc = extract_toc(&doc);
    assert!(toc.is_empty(), "Simple PDF should have no TOC");
}

#[test]
fn test_extract_toc_with_chapters() {
    let doc = create_pdf_with_toc();
    let toc = extract_toc(&doc);

    // Should have 2 top-level entries
    assert_eq!(toc.len(), 2, "Should have 2 top-level TOC entries");

    // Check first chapter
    assert_eq!(toc[0].title, "Chapter 1");
    assert_eq!(toc[0].page, Some(1));

    // Check first chapter has a child
    assert_eq!(toc[0].children.len(), 1, "Chapter 1 should have 1 child");
    assert_eq!(toc[0].children[0].title, "Section 1.1");
    assert_eq!(toc[0].children[0].page, Some(2));

    // Check second chapter
    assert_eq!(toc[1].title, "Chapter 2");
    assert_eq!(toc[1].page, Some(3));
    assert!(
        toc[1].children.is_empty(),
        "Chapter 2 should have no children"
    );
}

#[test]
fn test_pdf_with_utf16_metadata() {
    use pedaru_lib::encoding::decode_pdf_string;

    let doc = create_pdf_with_utf16_metadata();

    // Get Info dictionary
    if let Ok(Object::Reference(info_ref)) = doc.trailer.get(b"Info") {
        if let Ok(info_dict) = doc.get_dictionary(*info_ref) {
            // Test title decoding
            let title = info_dict.get(b"Title").ok().and_then(decode_pdf_string);
            assert_eq!(title, Some("Unicode Title".to_string()));

            // Test author decoding
            let author = info_dict.get(b"Author").ok().and_then(decode_pdf_string);
            assert_eq!(author, Some("Author Name".to_string()));
        } else {
            panic!("Failed to get Info dictionary");
        }
    } else {
        panic!("No Info reference in trailer");
    }
}

#[test]
fn test_pdf_file_roundtrip() {
    // Test that we can save and reload a PDF
    let mut doc = create_pdf_with_toc();
    let temp_file = save_to_temp_file(&mut doc);

    // Reload the document
    let reloaded = Document::load(temp_file.path()).expect("Failed to reload PDF");
    let toc = extract_toc(&reloaded);

    assert_eq!(toc.len(), 2, "Reloaded PDF should have 2 TOC entries");
    assert_eq!(toc[0].title, "Chapter 1");
}

#[test]
fn test_toc_entry_serialization() {
    let entry = TocEntry {
        title: "Test Chapter".to_string(),
        page: Some(5),
        children: vec![TocEntry {
            title: "Test Section".to_string(),
            page: Some(6),
            children: vec![],
        }],
    };

    // Serialize to JSON
    let json = serde_json::to_string(&entry).expect("Failed to serialize");
    assert!(json.contains("Test Chapter"));
    assert!(json.contains("Test Section"));
    assert!(json.contains("5"));
    assert!(json.contains("6"));
}

#[test]
fn test_create_multiple_page_pdf() {
    let doc = create_simple_pdf(10);
    let pages = doc.get_pages();
    assert_eq!(pages.len(), 10, "Should have 10 pages");
}

// ============================================================================
// Encrypted PDF tests (using fixture files from tests/fixtures/)
// ============================================================================

#[test]
fn test_encrypted_pdf_with_empty_password_loads() {
    // Test that lopdf can load a PDF encrypted with empty user password
    let doc = load_encrypted_pdf_fixture("encrypted_empty_password.pdf");

    // The document should have pages
    let pages = doc.get_pages();
    assert_eq!(pages.len(), 1, "Encrypted PDF should have 1 page");
}

#[test]
fn test_encrypted_pdf_has_encrypt_dict() {
    // Test that the encrypted PDF has an Encrypt dictionary
    let doc = load_encrypted_pdf_fixture("encrypted_empty_password.pdf");

    // Check for Encrypt entry in trailer
    let has_encrypt = doc.trailer.get(b"Encrypt").is_ok();
    assert!(has_encrypt, "Encrypted PDF should have Encrypt dictionary");
}

#[test]
fn test_encrypted_pdf_toc_extraction() {
    // Test TOC extraction from encrypted PDF (should work with empty password)
    let doc = load_encrypted_pdf_fixture("encrypted_empty_password.pdf");

    // This simple encrypted PDF has no TOC, so it should return empty
    let toc = extract_toc(&doc);
    assert!(
        toc.is_empty(),
        "Simple encrypted PDF should have no TOC entries"
    );
}

#[test]
fn test_encrypted_pdf_is_recognized_as_encrypted() {
    // Verify the document is properly identified as encrypted
    let doc = load_encrypted_pdf_fixture("encrypted_empty_password.pdf");

    // Check if encryption_state is set (indicates successful decryption)
    // The document has an Encrypt dict, meaning it was encrypted
    let encrypt_ref = doc.trailer.get(b"Encrypt");
    assert!(
        encrypt_ref.is_ok(),
        "Document should have encryption dictionary"
    );
}

// ============================================================================
// Encrypted PDF with English metadata and TOC tests (runtime-generated)
// ============================================================================

#[test]
fn test_encrypted_pdf_english_loads() {
    let doc = load_generated_encrypted_english_pdf();

    let pages = doc.get_pages();
    assert_eq!(
        pages.len(),
        3,
        "Encrypted PDF with English content should have 3 pages"
    );
    assert!(
        doc.trailer.get(b"Encrypt").is_ok(),
        "Generated encrypted PDF should have Encrypt dictionary"
    );
}

#[test]
fn test_encrypted_pdf_english_title_decodes_correctly() {
    use pedaru_lib::encoding::decode_pdf_string;

    let doc = load_generated_encrypted_english_pdf();
    if let Ok(Object::Reference(info_ref)) = doc.trailer.get(b"Info") {
        if let Ok(info_dict) = doc.get_dictionary(*info_ref) {
            let title = info_dict.get(b"Title").ok().and_then(decode_pdf_string);
            assert_eq!(
                title,
                Some("Encrypted Test Document".to_string()),
                "Title should be decoded exactly"
            );
        } else {
            panic!("Failed to get Info dictionary");
        }
    } else {
        panic!("No Info reference in trailer");
    }
}

#[test]
fn test_encrypted_pdf_english_author_decodes_correctly() {
    use pedaru_lib::encoding::decode_pdf_string;

    let doc = load_generated_encrypted_english_pdf();
    if let Ok(Object::Reference(info_ref)) = doc.trailer.get(b"Info") {
        if let Ok(info_dict) = doc.get_dictionary(*info_ref) {
            let author = info_dict.get(b"Author").ok().and_then(decode_pdf_string);
            assert_eq!(
                author,
                Some("Sample Author".to_string()),
                "Author should be decoded exactly"
            );
        } else {
            panic!("Failed to get Info dictionary");
        }
    } else {
        panic!("No Info reference in trailer");
    }
}

#[test]
fn test_encrypted_pdf_english_toc_extraction() {
    let doc = load_generated_encrypted_english_pdf();
    let toc = extract_toc(&doc);
    assert_eq!(
        toc.len(),
        2,
        "Should have 2 top-level TOC entries. Got: {:?}",
        toc
    );
}

#[test]
fn test_encrypted_pdf_english_toc_titles() {
    let doc = load_generated_encrypted_english_pdf();
    let toc = extract_toc(&doc);

    assert!(!toc.is_empty(), "TOC should not be empty");
    assert_eq!(
        toc[0].title, "Chapter 1 Introduction",
        "First TOC entry title mismatch"
    );
    assert_eq!(toc[0].page, Some(1), "First chapter should be on page 1");

    assert_eq!(
        toc[0].children.len(),
        1,
        "First chapter should have 1 child"
    );
    assert_eq!(
        toc[0].children[0].title, "Section 1.1 Overview",
        "Child TOC entry title mismatch"
    );
    assert_eq!(
        toc[0].children[0].page,
        Some(2),
        "Child section should be on page 2"
    );

    assert_eq!(
        toc[1].title, "Chapter 2 Main Discussion",
        "Second TOC entry title mismatch"
    );
    assert_eq!(toc[1].page, Some(3), "Second chapter should be on page 3");
    assert!(
        toc[1].children.is_empty(),
        "Second chapter should have no children"
    );
}

#[test]
fn test_encrypted_pdf_english_toc_page_numbers() {
    let doc = load_generated_encrypted_english_pdf();
    let toc = extract_toc(&doc);

    assert_eq!(toc.len(), 2, "Should have 2 top-level entries");
    assert_eq!(toc[0].page, Some(1), "Chapter 1 should be on page 1");
    assert_eq!(
        toc[0].children[0].page,
        Some(2),
        "Section 1.1 should be on page 2"
    );
    assert_eq!(toc[1].page, Some(3), "Chapter 2 should be on page 3");
}
