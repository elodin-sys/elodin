use elodin_types::api::{GenerateLicenseReq, LicenseType};
use miette::IntoDiagnostic;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use super::Cli;

const PUBLIC_KEY: &[u8] = &[
    51, 103, 111, 138, 222, 73, 65, 199, 75, 97, 94, 218, 95, 159, 191, 166, 82, 106, 48, 104, 17,
    136, 6, 75, 93, 37, 184, 82, 188, 194, 130, 20,
];

#[derive(Debug, Serialize, Deserialize)]
struct LicenseKey {
    machine_id: Vec<u8>,
    user_id: Uuid,
    expires_at: u64,
    license_type: LicenseType,
}

impl LicenseKey {
    fn from_bytes(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }

    fn verify(&self) -> Result<(), LicenseKeyError> {
        let id = machine_uid::get().unwrap();
        let id = id.as_bytes();
        if self.machine_id != id {
            return Err(LicenseKeyError::WrongMachine);
        }
        if SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            >= self.expires_at
        {
            return Err(LicenseKeyError::Expired);
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SignedLicenseKey {
    sig: Vec<u8>,
    license_key: Vec<u8>,
}

impl SignedLicenseKey {
    fn from_bytes(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }

    fn verify(&self) -> Result<LicenseKey, LicenseKeyError> {
        let public_key =
            ring::signature::UnparsedPublicKey::new(&ring::signature::ED25519, PUBLIC_KEY);
        if public_key.verify(&self.license_key, &self.sig).is_err() {
            return Err(LicenseKeyError::InvalidSignature);
        }
        let license_key = LicenseKey::from_bytes(&self.license_key)
            .into_diagnostic()
            .map_err(LicenseKeyError::Other)?;
        license_key.verify()?;
        Ok(license_key)
    }
}

impl Cli {
    pub async fn verify_license_key(&self) -> Result<(), LicenseKeyError> {
        match self.license_key().await {
            Ok(key) if key.license_type == LicenseType::None => Err(LicenseKeyError::NoLicenseKey),
            Ok(_) => Ok(()),
            Err(err) => Err(err),
        }
    }

    async fn license_key(&self) -> Result<LicenseKey, LicenseKeyError> {
        let license_key = match self.license_key_inner() {
            Ok(license_key) if license_key.license_type == LicenseType::None => {
                self.pull_license_key()
                    .await
                    .map_err(LicenseKeyError::Other)?;
                self.license_key_inner()?
            }
            Err(_) => {
                self.pull_license_key()
                    .await
                    .map_err(LicenseKeyError::Other)?;
                self.license_key_inner()?
            }
            Ok(license_key) => return Ok(license_key),
        };
        // pull new license key in background
        let this = self.clone();
        tokio::spawn(async move {
            let _ = this.pull_license_key().await;
        });
        Ok(license_key)
    }

    fn license_key_inner(&self) -> Result<LicenseKey, LicenseKeyError> {
        let dirs = self
            .dirs()
            .into_diagnostic()
            .map_err(LicenseKeyError::Other)?;
        let data_dir = dirs.data_dir();
        let license_key = data_dir.join("license_key");
        if !license_key.exists() {
            return Err(LicenseKeyError::KeyFileNotFound);
        }
        let license_key = std::fs::read(license_key)
            .into_diagnostic()
            .map_err(LicenseKeyError::Other)?;
        let signed_license_key = SignedLicenseKey::from_bytes(&license_key)
            .into_diagnostic()
            .map_err(LicenseKeyError::Other)?;
        signed_license_key.verify()
    }

    async fn pull_license_key(&self) -> miette::Result<()> {
        let mut client = self.client().await?;
        let id = machine_uid::get().unwrap();
        let resp = client
            .generate_license(GenerateLicenseReq {
                machine_id: id.as_bytes().to_vec(),
            })
            .await
            .into_diagnostic()?
            .into_inner();
        let dirs = self.dirs().into_diagnostic()?;
        let data_dir = dirs.data_dir();
        let license_key = data_dir.join("license_key");
        std::fs::write(license_key, resp.license).into_diagnostic()?;
        Ok(())
    }
}

#[derive(miette::Diagnostic, thiserror::Error, Debug)]
pub enum LicenseKeyError {
    #[error(
        "Your account does not have an Elodin license. Please purchase on before using the CLI"
    )]
    #[diagnostic(code(elodin::no_license), url("https://www.elodin.systems/pricing"))]
    NoLicenseKey,
    #[error("Your license key is not valid for this machine")]
    WrongMachine,
    #[error("Your license key has an invalid signature")]
    InvalidSignature,
    #[error("Your license key is expired, and we couldn't pull a new one. Your computer may be offline.")]
    Expired,
    #[error("Your license key file could not be found")]
    KeyFileNotFound,
    #[error("{0}")]
    #[diagnostic(transparent)]
    Other(miette::Error),
}

// impl<E: Into<miette::Error>> From<E> for LicenseKeyError {
//     fn from(value: E) -> Self {
//         LicenseKeyError::Other(value.into())
//     }
// }
