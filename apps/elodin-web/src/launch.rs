use elodin_editor::ViewerCapabilities;

pub fn capabilities(values: &[String]) -> Result<ViewerCapabilities, String> {
    match values {
        [] => Ok(ViewerCapabilities::default()),
        [value] if value == "false" => Ok(ViewerCapabilities::default()),
        [value] if value == "true" => Ok(ViewerCapabilities::READ_ONLY),
        [_] => Err("read_only must be true or false".into()),
        _ => Err("Duplicate read_only parameter".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_only_capabilities_are_explicit() {
        assert_eq!(capabilities(&[]).unwrap(), ViewerCapabilities::default());
        assert_eq!(
            capabilities(&["true".into()]).unwrap(),
            ViewerCapabilities::READ_ONLY
        );
        assert_eq!(
            capabilities(&["false".into()]).unwrap(),
            ViewerCapabilities::default()
        );
        assert!(capabilities(&["yes".into()]).is_err());
        assert!(capabilities(&["true".into(), "false".into()]).is_err());
    }
}
