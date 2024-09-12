use super::{Claims, UserInfo};
use crate::error::Error;
use futures::Future;
use jsonwebtoken::{
    decode, decode_header,
    jwk::{AlgorithmParameters, JwkSet},
    Algorithm, DecodingKey, Validation,
};
use reqwest::Client;
use std::str::FromStr;
use tonic::{Request, Response, Status};
use tracing::warn;

impl super::Api {
    pub async fn authed_route<Req, Resp, RespFuture>(
        &self,
        req: Request<Req>,
        handler: impl FnOnce(Req, Option<Claims>) -> RespFuture,
    ) -> Result<tonic::Response<Resp>, Status>
    where
        RespFuture: Future<Output = Result<Resp, Error>>,
    {
        let handle = |req: Request<Req>, claims| async move {
            handler(req.into_inner(), claims)
                .await
                .map_err(Error::status)
                .map(Response::new)
        };
        let Some(auth_header) = req.metadata().get("Authorization") else {
            return handle(req, None).await;
        };

        let Ok(auth_header) = auth_header.to_str() else {
            return handle(req, None).await;
        };
        let Some(token) = auth_header.split("Bearer ").nth(1) else {
            return handle(req, None).await;
        };
        let Ok(claims) = validate_auth_header(
            token,
            &self.auth_context.auth_config.domain,
            &self.auth_context.auth0_keys,
        ) else {
            return handle(req, None).await;
        };
        handle(req, Some(claims)).await
    }

    pub async fn authed_route_userinfo<Req, Resp, RespFuture>(
        &self,
        req: Request<Req>,
        handler: impl FnOnce(Req, UserInfo) -> RespFuture,
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
        let req_client = reqwest::Client::new();

        let userinfo =
            get_userinfo(&req_client, &self.auth_context.auth_config.domain, token).await?;

        handler(req.into_inner(), userinfo)
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
                let claims = claims.ok_or(Error::Unauthorized)?;
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
                let claims = claims.ok_or(Error::Unauthorized)?;
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

#[macro_export]
macro_rules! optional_current_user_route {
    ($self:ident, $req:ident, $handler:expr) => {
        $self
            .authed_route($req, move |req, claims| async move {
                if let Some(claims) = claims {
                    let user = atc_entity::User::find()
                        .filter(atc_entity::user::Column::Auth0Id.eq(&claims.sub))
                        .one(&$self.db)
                        .await?
                        .ok_or_else(|| Error::Unauthorized)?;
                    $handler($self, req, Some(CurrentUser { user, claims })).await
                } else {
                    $handler($self, req, None).await
                }
            })
            .await
    };
}

#[macro_export]
macro_rules! optional_current_user_route_txn {
    ($self:ident, $req:ident, $handler:expr) => {
        $self
            .authed_route($req, move |req, claims| async move {
                let txn = $self.db.begin().await?;
                let user = if let Some(claims) = claims {
                    let user = atc_entity::User::find()
                        .filter(atc_entity::user::Column::Auth0Id.eq(&claims.sub))
                        .one(&$self.db)
                        .await?
                        .ok_or_else(|| Error::Unauthorized)?;
                    Some(CurrentUser { user, claims })
                } else {
                    None
                };
                let res = $handler($self, req, user, &txn).await;
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

pub async fn get_userinfo(
    client: &Client,
    domain: &str,
    access_token: &str,
) -> Result<UserInfo, Error> {
    client
        .get(format!("https://{}/userinfo", domain))
        .bearer_auth(access_token)
        .send()
        .await?
        .json::<UserInfo>()
        .await
        .map_err(Error::from)
}

pub async fn get_keyset(domain: &str) -> Result<JwkSet, Error> {
    reqwest::get(&format!("https://{}/.well-known/jwks.json", domain))
        .await?
        .json()
        .await
        .map_err(Error::from)
}

pub fn validate_auth_header(
    token: &str,
    domain: &str,
    auth0_keys: &JwkSet,
) -> Result<Claims, Error> {
    let header = decode_header(token).map_err(|_| Error::Unauthorized)?;
    let kid = header.kid.ok_or(Error::Unauthorized)?;
    let Some(j) = auth0_keys.find(&kid) else {
        return Err(Error::Unauthorized);
    };

    let AlgorithmParameters::RSA(rsa) = &j.algorithm else {
        return Err(Error::Unauthorized);
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
    validation.set_audience(&[format!("https://{domain}/atc")]);
    let claims =
        decode::<Claims>(token, &decoding_key, &validation).map_err(|_| Error::Unauthorized)?;
    Ok(claims.claims)
}
