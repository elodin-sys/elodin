use super::*;

/// Bevy resolves relative glTF URIs as paths. That drops the parent's query
/// from both the request and the asset cache identity. Give dependencies full
/// session URLs before the glTF loader sees them, including GLB JSON chunks.
pub(super) fn prepare(
    bytes: Vec<u8>,
    url: &str,
    endpoint: Option<&DbEndpoint>,
) -> Result<Vec<u8>, AssetReaderError> {
    let Some(endpoint @ DbEndpoint::Browser(_)) = endpoint else {
        return Ok(bytes);
    };
    let path = url.split('?').next().unwrap_or(url).to_ascii_lowercase();
    let glb = path.ends_with(".glb");
    if !glb && !path.ends_with(".gltf") {
        return Ok(bytes);
    }
    let invalid = |message: &str| {
        AssetReaderError::Io(io::Error::new(io::ErrorKind::InvalidData, message.to_owned()).into())
    };
    let json = if glb {
        if bytes.len() < 20
            || &bytes[..4] != b"glTF"
            || bytes[4..8] != 2u32.to_le_bytes()
            || bytes[16..20] != 0x4e4f534au32.to_le_bytes()
        {
            return Err(invalid("Invalid GLB header"));
        }
        let len = u32::from_le_bytes(bytes[12..16].try_into().expect("header checked")) as usize;
        bytes
            .get(20..20usize.saturating_add(len))
            .ok_or_else(|| invalid("Invalid GLB JSON chunk"))?
    } else {
        &bytes
    };
    let mut document: serde_json::Value =
        serde_json::from_slice(json).map_err(|e| invalid(&e.to_string()))?;
    let mut changed = false;
    for collection in ["buffers", "images"] {
        if let Some(items) = document.get_mut(collection).and_then(|v| v.as_array_mut()) {
            for item in items {
                let Some(uri) = item.get("uri").and_then(|v| v.as_str()) else {
                    continue;
                };
                if uri.starts_with("data:") {
                    continue;
                }
                let resolved = endpoint
                    .asset_dependency_url(url, uri)
                    .map_err(|e| invalid(&e))?;
                // The Bevy glTF loader percent-decodes URIs once before it
                // parses AssetPath. Preserve escaped '#' and query characters.
                item["uri"] = resolved.replace('%', "%25").into();
                changed = true;
            }
        }
    }
    if !changed {
        return Ok(bytes);
    }
    let mut rewritten = serde_json::to_vec(&document).map_err(|e| invalid(&e.to_string()))?;
    if !glb {
        return Ok(rewritten);
    }
    while !rewritten.len().is_multiple_of(4) {
        rewritten.push(b' ');
    }
    let old_len = u32::from_le_bytes(bytes[12..16].try_into().expect("header checked")) as usize;
    let tail = &bytes[20 + old_len..];
    let total = u32::try_from(20 + rewritten.len() + tail.len())
        .map_err(|_| invalid("GLB exceeds the format size limit"))?;
    let mut output = Vec::with_capacity(total as usize);
    output.extend_from_slice(&bytes[..8]);
    output.extend_from_slice(&total.to_le_bytes());
    output.extend_from_slice(&(rewritten.len() as u32).to_le_bytes());
    output.extend_from_slice(&bytes[16..20]);
    output.extend_from_slice(&rewritten);
    output.extend_from_slice(tail);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn endpoint(route: &str) -> DbEndpoint {
        impeller_bevy::BrowserEndpoints::new(
            "wss://example.com/main",
            "wss://example.com/messages",
            &format!("https://example.com/assets?route={route}"),
        )
        .unwrap()
        .into()
    }
    fn model() -> Vec<u8> {
        br##"{"asset":{"version":"2.0"},"buffers":[{"uri":"buffer.bin","byteLength":4}],"images":[{"uri":"textures/a%23b.png"},{"uri":"data:image/png;base64,AA=="}]}"##.to_vec()
    }
    #[test]
    fn gltf_dependencies_keep_queries_in_bevy_cache_keys() {
        let first = prepare(
            model(),
            "https://example.com/assets/model.gltf?route=first",
            Some(&endpoint("first")),
        )
        .unwrap();
        let second = prepare(
            model(),
            "https://example.com/assets/model.gltf?route=second",
            Some(&endpoint("second")),
        )
        .unwrap();
        let first: serde_json::Value = serde_json::from_slice(&first).unwrap();
        let second: serde_json::Value = serde_json::from_slice(&second).unwrap();
        assert_eq!(
            first["images"][0]["uri"],
            "https://example.com/assets/textures/a%2523b.png?route=first"
        );
        assert_ne!(first["buffers"][0]["uri"], second["buffers"][0]["uri"]);
        assert_eq!(first["images"][1]["uri"], "data:image/png;base64,AA==");
        let decoded = first["images"][0]["uri"]
            .as_str()
            .unwrap()
            .replace("%25", "%");
        let dependency =
            bevy::asset::AssetPath::parse("https://example.com/assets/model.gltf?route=first")
                .resolve_embed_str(&decoded)
                .unwrap();
        assert_eq!(
            dependency.to_string(),
            "https://example.com/assets/textures/a%23b.png?route=first"
        );
        assert_eq!(dependency.get_extension(), Some("png"));
    }
    #[test]
    fn glb_keeps_binary_chunks_and_updates_lengths() {
        let mut json = model();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let tail = [4, 0, 0, 0, b'B', b'I', b'N', 0, 1, 2, 3, 4];
        let mut glb = b"glTF".to_vec();
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&((20 + json.len() + tail.len()) as u32).to_le_bytes());
        glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x4e4f534au32.to_le_bytes());
        glb.extend_from_slice(&json);
        glb.extend_from_slice(&tail);
        let prepared = prepare(
            glb.clone(),
            "https://example.com/assets/model.glb?route=one",
            Some(&endpoint("one")),
        )
        .unwrap();
        assert_eq!(
            u32::from_le_bytes(prepared[8..12].try_into().unwrap()) as usize,
            prepared.len()
        );
        assert!(prepared.ends_with(&tail));
        assert_eq!(
            prepare(glb.clone(), "http://localhost/model.glb", None).unwrap(),
            glb
        );
        assert!(
            prepare(
                b"glTF".to_vec(),
                "https://example.com/model.glb",
                Some(&endpoint("one"))
            )
            .is_err()
        );
    }
}
