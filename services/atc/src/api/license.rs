use std::time::SystemTime;

use elodin_types::api::LicenseType;
use elodin_types::api::{GenerateLicenseReq, GenerateLicenseResp};
use ring::signature::Ed25519KeyPair;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{Api, CurrentUser};
use crate::error::Error;

const PRIV_KEY: &[u8] = &[
    48, 81, 2, 1, 1, 48, 5, 6, 3, 43, 101, 112, 4, 34, 4, 32, 127, 217, 116, 108, 120, 120, 183,
    34, 178, 49, 191, 27, 133, 8, 174, 7, 169, 136, 138, 172, 104, 140, 68, 184, 94, 73, 58, 105,
    221, 68, 69, 159, 129, 33, 0, 51, 103, 111, 138, 222, 73, 65, 199, 75, 97, 94, 218, 95, 159,
    191, 166, 82, 106, 48, 104, 17, 136, 6, 75, 93, 37, 184, 82, 188, 194, 130, 20,
];

impl Api {
    pub async fn generate_license(
        &self,
        req: GenerateLicenseReq,
        current_user: CurrentUser,
    ) -> Result<GenerateLicenseResp, Error> {
        let key_pair = ring::signature::Ed25519KeyPair::from_pkcs8(PRIV_KEY).expect("invalid key");
        let expires_at = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3600 * 24 * 30;

        let license_key = LicenseKey {
            machine_id: req.machine_id,
            user_id: current_user.user.id,
            expires_at,
            license_type: current_user.user.license_type.into(),
        }
        .sign(&key_pair);
        let license = postcard::to_allocvec(&license_key)?;
        Ok(GenerateLicenseResp { license })
    }
}

impl LicenseKey {
    fn sign(&self, key: &Ed25519KeyPair) -> SignedLicenseKey {
        let license_key = postcard::to_allocvec(self).unwrap();
        let sig = key.sign(&license_key);
        let sig = sig.as_ref().to_vec();
        SignedLicenseKey { sig, license_key }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct LicenseKey {
    machine_id: Vec<u8>,
    user_id: Uuid,
    expires_at: u64,
    license_type: LicenseType,
}

#[derive(Debug, Serialize, Deserialize)]
struct SignedLicenseKey {
    sig: Vec<u8>,
    license_key: Vec<u8>,
}
