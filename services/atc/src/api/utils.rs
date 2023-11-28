use super::Claims;
use crate::error::Error;
use futures::Future;
use jsonwebtoken::{
    decode, decode_header,
    jwk::{AlgorithmParameters, JwkSet},
    Algorithm, DecodingKey, Validation,
};
use std::str::FromStr;
use tonic::{Request, Response, Status};
use tracing::warn;

impl super::Api {
    pub async fn authed_route<Req, Resp, RespFuture>(
        &self,
        req: Request<Req>,
        handler: impl FnOnce(Req, Claims) -> RespFuture,
    ) -> Result<tonic::Response<Resp>, Status>
    where
        RespFuture: Future<Output = Result<Resp, Error>>,
    {
        let auth_header = req
            .metadata()
            .get("Authorization")
            .ok_or(Error::Unauthorized)?;
        let auth_header = auth_header.to_str().map_err(|_| Error::Unauthorized)?;
        let token = auth_header
            .split("Bearer ")
            .nth(1)
            .ok_or(Error::Unauthorized)?;
        let header = decode_header(token).map_err(|_| Error::Unauthorized)?;
        let kid = header.kid.ok_or(Error::Unauthorized)?;
        let Some(j) = self.auth0_keys.find(&kid) else {
            return Err(Error::Unauthorized.status());
        };

        let AlgorithmParameters::RSA(rsa) = &j.algorithm else {
            return Err(Error::Unauthorized.into());
        };

        let decoding_key = DecodingKey::from_rsa_components(&rsa.n, &rsa.e).unwrap();

        let mut validation = Validation::new(
            Algorithm::from_str(
                j.common
                    .key_algorithm
                    .ok_or_else(|| {
                        warn!("missing key algo field in jwks");
                        Error::Unauthorized
                    })?
                    .to_string()
                    .as_str(),
            )
            .map_err(|_| {
                warn!("invalid jwks algo");
                Error::Unauthorized
            })?,
        );
        validation.validate_exp = true;
        validation.set_audience(&[&self.config.auth0.client_id]);
        let claims =
            decode::<Claims>(token, &decoding_key, &validation).map_err(|_| Error::Unauthorized)?;

        handler(req.into_inner(), claims.claims)
            .await
            .map_err(Error::status)
            .map(Response::new)
    }
}

#[macro_export]
macro_rules! current_user_route_txn {
    ($self:ident, $req:ident, $handler:expr) => {
        $self
            .authed_route($req, move |req, claims| async move {
                let txn = $self.db.begin().await?;
                let user = atc_entity::User::find()
                    .filter(atc_entity::user::Column::Auth0Id.eq(&claims.sub))
                    .one(&txn)
                    .await?
                    .ok_or_else(|| Error::Unauthorized)?;
                let res = $handler($self, req, CurrentUser { user, claims }, &txn).await;
                if res.is_ok() {
                    txn.commit().await?;
                } else {
                    txn.rollback().await?;
                }
                res
            })
            .await
    };
}

#[macro_export]
macro_rules! current_user_route {
    ($self:ident, $req:ident, $handler:expr) => {
        $self
            .authed_route($req, move |req, claims| async move {
                let user = atc_entity::User::find()
                    .filter(atc_entity::user::Column::Auth0Id.eq(&claims.sub))
                    .one(&$self.db)
                    .await?
                    .ok_or_else(|| Error::Unauthorized)?;
                $handler($self, req, CurrentUser { user, claims }).await
            })
            .await
    };
}

#[allow(dead_code)]
pub async fn route<Req, Resp, RespFuture>(
    req: Request<Req>,
    handler: impl Fn(Req) -> RespFuture,
) -> Result<tonic::Response<Resp>, Status>
where
    RespFuture: Future<Output = Result<Resp, Error>>,
{
    handler(req.into_inner())
        .await
        .map_err(Error::status)
        .map(Response::new)
}

pub async fn get_keyset(domain: &str) -> Result<JwkSet, Error> {
    reqwest::get(&format!("https://{}/.well-known/jwks.json", domain))
        .await?
        .json()
        .await
        .map_err(Error::from)
}
