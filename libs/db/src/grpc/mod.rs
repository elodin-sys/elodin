pub mod v1 {
    tonic::include_proto!("elodin.db.v1");
}

mod ingest;

pub use ingest::serve;

#[cfg(test)]
mod tests {
    use super::v1::{AckPolicy, SchemaSet, SessionOpen};
    use prost::Message;

    #[test]
    fn generated_message_round_trip() {
        let open = SessionOpen {
            client_name: "test-client".into(),
            schema_fingerprint: vec![1, 2, 3],
            schema: Some(SchemaSet::default()),
            ack_policy: Some(AckPolicy {
                max_unacked_rows: 256,
                max_ack_delay_ms: 100,
            }),
            client_instance_id: vec![4, 5, 6],
        };

        let decoded = SessionOpen::decode(open.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decoded, open);
    }
}
