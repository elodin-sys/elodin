//! HTTP helpers: blocking reqwest on native, `fetch` on wasm.

pub async fn get_text(url: &str) -> Result<String, String> {
    let bytes = get_bytes(url).await?;
    String::from_utf8(bytes).map_err(|err| format!("{url}: {err}"))
}

pub async fn get_bytes(url: &str) -> Result<Vec<u8>, String> {
    let (status, body) = request("GET", url, None).await?;
    if !(200..300).contains(&status) {
        return Err(format!("{url}: HTTP {status}"));
    }
    Ok(body)
}

pub async fn put_bytes(url: &str, bytes: Vec<u8>) -> Result<(), String> {
    let (status, _) = request("PUT", url, Some(bytes)).await?;
    if !(200..300).contains(&status) {
        return Err(format!("{url}: HTTP {status}"));
    }
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
async fn request(method: &str, url: &str, body: Option<Vec<u8>>) -> Result<(u16, Vec<u8>), String> {
    let timeout = if method == "PUT" {
        std::time::Duration::from_secs(30)
    } else {
        std::time::Duration::from_secs(5)
    };
    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|err| format!("{url}: {err}"))?;
    let mut builder = match method {
        "PUT" => client.put(url),
        _ => client.get(url),
    };
    if let Some(body) = body {
        builder = builder.body(body);
    }
    let response = builder.send().map_err(|err| format!("{url}: {err}"))?;
    let status = response.status().as_u16();
    let bytes = response.bytes().map_err(|err| format!("{url}: {err}"))?;
    Ok((status, bytes.to_vec()))
}

#[cfg(target_family = "wasm")]
async fn request(method: &str, url: &str, body: Option<Vec<u8>>) -> Result<(u16, Vec<u8>), String> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, RequestMode, Response};

    let window = web_sys::window().ok_or_else(|| format!("{url}: no window"))?;
    let opts = RequestInit::new();
    opts.set_method(method);
    opts.set_mode(RequestMode::Cors);
    if let Some(bytes) = body {
        let array = js_sys::Uint8Array::from(bytes.as_slice());
        opts.set_body(&array);
    }
    let request =
        Request::new_with_str_and_init(url, &opts).map_err(|err| format!("{url}: {err:?}"))?;
    let resp_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|err| format!("{url}: {err:?}"))?;
    let response: Response = resp_value
        .dyn_into()
        .map_err(|err| format!("{url}: {err:?}"))?;
    let status = response.status();
    let buf = JsFuture::from(
        response
            .array_buffer()
            .map_err(|err| format!("{url}: {err:?}"))?,
    )
    .await
    .map_err(|err| format!("{url}: {err:?}"))?;
    let array = js_sys::Uint8Array::new(&buf);
    let mut bytes = vec![0u8; array.length() as usize];
    array.copy_to(&mut bytes);
    Ok((status, bytes))
}
