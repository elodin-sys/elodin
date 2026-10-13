use elodin_mcp::index::Index;
use elodin_mcp::search::{MAX_OUTPUT, search};
use elodin_mcp::source::LocalDb;
use serde_json::Value;

fn fixture_index() -> (tempfile::TempDir, Index) {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("flight");
    elodin_db::fixture::write(&path).unwrap();
    let run = LocalDb::parse(path.to_str().unwrap())
        .unwrap()
        .run()
        .unwrap();
    (tmp, Index::new(vec![run]))
}

fn names(v: &Value) -> Vec<String> {
    v["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn searches_the_fixture() {
    let (_tmp, index) = fixture_index();

    let overview = search(&index, None, None);
    let tables: Vec<_> = overview["results"][1]["tables"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(tables, ["baro", "gps", "imu"]);
    assert_eq!(overview["results"][0]["runs"][0]["name"], "flight");

    let accel = search(&index, Some("signals('accel')"), None);
    assert_eq!(names(&accel)[0], "imu.accel");
    assert_eq!(accel["results"][0]["sdk"], "el.signal('imu.accel')");
    assert_eq!(accel["results"][0]["runs"]["flight"]["rows"], 6001);
    assert_eq!(names(&search(&index, Some("acel"), None))[0], "imu.accel");
    assert_eq!(names(&search(&index, Some("tables('gp')"), None)), ["gps"]);

    let gone = search(&index, Some("describe('secret.key')"), None);
    assert_eq!(gone["results"].as_array().unwrap().len(), 0);
    assert!(
        !serde_json::to_string(&search(&index, Some("signals()"), Some(100)))
            .unwrap()
            .contains("secret")
    );

    let err = search(&index, Some("signals(accel)"), None);
    assert!(err["grammar"].is_string());
    let missing = search(&index, Some("describe('imu.acel')"), None);
    assert_eq!(missing["did_you_mean"][0], "imu.accel");

    for code in [None, Some("signals()"), Some("sdk()")] {
        let out = serde_json::to_string(&search(&index, code, Some(1000))).unwrap();
        assert!(out.len() <= MAX_OUTPUT, "{code:?}: {}", out.len());
    }
}
