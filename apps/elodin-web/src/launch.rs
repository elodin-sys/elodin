use elodin_editor::{BrowserEndpoints, DbEndpoint, ViewerCapabilities};

pub struct Launch {
    pub endpoint: Option<DbEndpoint>,
    pub capabilities: ViewerCapabilities,
}

/// Browser routes are supplied together. `db` retains the standalone host:port mode.
pub fn parse(values: &[(String, String)]) -> Result<Launch, String> {
    let get = |name: &str| -> Result<Option<&str>, String> {
        let mut matches = values.iter().filter(|(key, _)| key == name);
        let value = matches.next().map(|(_, value)| value.as_str());
        if matches.next().is_some() {
            return Err(format!("Duplicate {name} parameter"));
        }
        Ok(value)
    };
    let impeller = get("impeller")?;
    let messages = get("messages")?;
    let assets = get("assets")?;
    let db = get("db")?;
    let browser = impeller.is_some() || messages.is_some() || assets.is_some();
    let endpoint = if browser {
        if db.is_some() {
            return Err("Use either db or browser endpoints".into());
        }
        Some(
            BrowserEndpoints::new(
                impeller.ok_or("Missing impeller endpoint")?,
                messages.ok_or("Missing messages endpoint")?,
                assets.ok_or("Missing assets endpoint")?,
            )?
            .into(),
        )
    } else {
        db.map(DbEndpoint::parse)
            .transpose()
            .map_err(|_| "Invalid db address")?
    };
    let read_only = match get("read_only")? {
        None => browser,
        Some("true") => true,
        Some("false") => false,
        Some(_) => return Err("read_only must be true or false".into()),
    };
    Ok(Launch {
        endpoint,
        capabilities: if read_only {
            ViewerCapabilities::READ_ONLY
        } else {
            ViewerCapabilities::default()
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn values(items: &[(&str, &str)]) -> Vec<(String, String)> {
        items
            .iter()
            .map(|(k, v)| ((*k).into(), (*v).into()))
            .collect()
    }
    #[test]
    fn rejects_incomplete_invalid_or_ambiguous_routes() {
        for items in [
            vec![("impeller", "wss://example.com/main")],
            vec![("db", "bad")],
            vec![("db", "localhost:2240"), ("db", "other:2240")],
            vec![("read_only", "yes")],
        ] {
            assert!(parse(&values(&items)).is_err());
        }
    }
    #[test]
    fn full_browser_routes_default_to_read_only() {
        let launch = parse(&values(&[
            ("impeller", "wss://example.com/s/main?route=a"),
            ("messages", "wss://example.com/s/msg?route=b"),
            ("assets", "https://example.com/s/assets?route=c"),
        ]))
        .unwrap();
        assert_eq!(launch.capabilities, ViewerCapabilities::READ_ONLY);
        assert_eq!(
            launch.endpoint.unwrap().ws_urls().1,
            "wss://example.com/s/msg?route=b"
        );
    }
    #[test]
    fn standalone_db_retains_desktop_style_urls() {
        let launch = parse(&values(&[("db", "localhost:2240")])).unwrap();
        assert_eq!(launch.capabilities, ViewerCapabilities::default());
        assert_eq!(
            launch.endpoint.unwrap().http_base(),
            "http://localhost:2241"
        );
        assert!(parse(&[]).unwrap().endpoint.is_none());
    }
}
