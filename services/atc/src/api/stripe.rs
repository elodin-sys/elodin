use crate::{
    config::{StripePlanConfig, StripePlansConfig},
    error::Error,
};
use atc_entity::{billing_account, events::DbExt, user};
use axum::{
    async_trait,
    extract::{FromRequest, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use chrono::{DateTime, Utc};
use elodin_types::api::{GetStripeSubscriptionStatusReq, LicenseType, StripeSubscriptionStatus};
use sea_orm::{
    ColumnTrait, ConnectionTrait, DatabaseTransaction, EntityTrait, FromQueryResult, QueryFilter,
    Set, Unchanged,
};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use stripe::{Customer, CustomerId, Event, EventObject, EventType, PriceId};
use tracing::Instrument;
use uuid::Uuid;

use super::{Api, AxumContext, CurrentUser};

pub struct StripeEvent(Event);

#[async_trait]
impl FromRequest<AxumContext> for StripeEvent {
    type Rejection = Response;

    async fn from_request(req: Request, state: &AxumContext) -> Result<Self, Self::Rejection> {
        let signature = if let Some(sig) = req.headers().get("stripe-signature") {
            sig.to_owned()
        } else {
            return Err(StatusCode::BAD_REQUEST.into_response());
        };

        let payload = String::from_request(req, state)
            .await
            .map_err(IntoResponse::into_response)?;

        Ok(Self(
            stripe::Webhook::construct_event(
                &payload,
                signature.to_str().unwrap(),
                &state.webhook_secret,
            )
            .map_err(|_| StatusCode::BAD_REQUEST.into_response())?,
        ))
    }
}

pub async fn stripe_webhook(
    State(context): State<AxumContext>,
    StripeEvent(event): StripeEvent,
) -> Result<impl IntoResponse, Error> {
    if let EventType::CustomerSubscriptionPaused
    | EventType::CustomerSubscriptionDeleted
    | EventType::CustomerSubscriptionCreated
    | EventType::CustomerSubscriptionUpdated
    | EventType::CustomerSubscriptionResumed
    | EventType::CustomerSubscriptionTrialWillEnd
    | EventType::CustomerSubscriptionPendingUpdateExpired
    | EventType::CustomerSubscriptionPendingUpdateApplied = event.type_
    {
        if let EventObject::Subscription(sub) = event.data.object {
            let Some(billing_account_id) =
                sub.metadata.get("billing_account_id").map(|s| s.as_str())
            else {
                tracing::warn!(
                    subscription_id = ?sub.id,
                    "no billing_account_id in subscription metadata",
                );
                return Ok(());
            };
            let billing_account_id = Uuid::parse_str(billing_account_id)?;
            return match sub.metadata.get("sub_type").as_ref().map(|s| s.as_str()) {
                Some("seat") => {
                    let mut license_type = LicenseType::None;
                    let mut seat_count = 0;
                    for item in &sub.items.data {
                        let Some(ref price) = item.price else {
                            continue;
                        };
                        let price_id = price.id.to_string();
                        let new_license_type = match price_id.as_str() {
                            id if id
                                == context.stripe_plans_config.commercial.subscription_price =>
                            {
                                LicenseType::Commercial
                            }
                            id if id
                                == context
                                    .stripe_plans_config
                                    .non_commercial
                                    .subscription_price =>
                            {
                                LicenseType::NonCommercial
                            }
                            id if id
                                == context.stripe_plans_config.commercial.monte_carlo_price =>
                            {
                                tracing::warn!(
                                    ?billing_account_id,
                                    price_id = id,
                                    "found commercial monte carlo price in seat subscription"
                                );
                                continue;
                            }
                            id if id
                                == context.stripe_plans_config.non_commercial.monte_carlo_price =>
                            {
                                tracing::warn!(
                                    ?billing_account_id,
                                    price_id = id,
                                    "found non_commercial monte carlo price in seat subscription"
                                );
                                continue;
                            }
                            price_id => {
                                tracing::warn!(
                                    ?billing_account_id,
                                    ?price_id,
                                    "unknown price id in seat subscription"
                                );
                                continue;
                            }
                        };
                        if license_type == LicenseType::None {
                            license_type = new_license_type;
                        } else if license_type != new_license_type {
                            tracing::warn!(
                                ?billing_account_id,
                                ?license_type,
                                ?new_license_type,
                                "multiple seat types in one subscription is unsupported"
                            );
                            continue;
                        };
                        seat_count += item.quantity.unwrap_or_default();
                    }
                    let Some(billing_account) =
                        billing_account::Entity::find_by_id(billing_account_id)
                            .one(&context.db)
                            .await?
                    else {
                        return Ok(());
                    };
                    let (license_type, seat_count) = if sub_status_active(&sub.status) {
                        (license_type, seat_count)
                    } else {
                        (LicenseType::None, 0)
                    };
                    billing_account::ActiveModel {
                        id: Unchanged(billing_account.id),
                        seat_subscription_id: Set(Some(sub.id.to_string())),
                        seat_count: Set(seat_count as i32),
                        seat_license_type: Set(license_type.into()),
                        ..Default::default()
                    }
                    .update_with_event(&context.db, &context.redis)
                    .await?;
                    sync_user_license(billing_account.owner_user_id, &context).await?;
                    Ok(())
                }
                Some("monte-carlo") => {
                    let monte_carlo_present = sub.items.data.iter().any(|item| {
                        let sub_price_id = item.price.as_ref().map(|price| price.id.as_str());
                        sub_price_id
                            == Some(&context.stripe_plans_config.commercial.monte_carlo_price)
                            || sub_price_id
                                == Some(
                                    &context.stripe_plans_config.non_commercial.monte_carlo_price,
                                )
                    });
                    let Some(billing_account) =
                        billing_account::Entity::find_by_id(billing_account_id)
                            .one(&context.db)
                            .await?
                    else {
                        return Ok(());
                    };
                    let monte_carlo_active = monte_carlo_present && sub_status_active(&sub.status);
                    billing_account::ActiveModel {
                        id: Unchanged(billing_account.id),
                        usage_subscription_id: Set(Some(sub.id.to_string())),
                        monte_carlo_active: Set(monte_carlo_active),
                        ..Default::default()
                    }
                    .update_with_event(&context.db, &context.redis)
                    .await?;
                    sync_user_license(billing_account.owner_user_id, &context).await?;
                    Ok(())
                }
                sub_type => {
                    tracing::warn!(?billing_account_id, ?sub_type, "unknown subscription type");
                    Ok(())
                }
            };
        }
    }
    Ok(())
}

async fn sync_user_license(user_id: Uuid, context: &AxumContext) -> Result<(), Error> {
    let billing_accounts = billing_account::Entity::find()
        .filter(billing_account::Column::OwnerUserId.eq(user_id))
        .all(&context.db)
        .await?;
    let (license_type, monte_carlo_active) = billing_accounts.into_iter().fold(
        (LicenseType::None, false),
        |(seat, monte_carlo), account| {
            let new_license_type = (account.seat_license_type as i32).max(seat as i32);
            let new_monte_carlo = account.monte_carlo_active || monte_carlo;
            (
                LicenseType::try_from(new_license_type).expect("incorrect license type"),
                new_monte_carlo,
            )
        },
    );
    user::ActiveModel {
        id: Unchanged(user_id),
        license_type: Set(license_type.into()),
        monte_carlo_active: Set(monte_carlo_active),
        ..Default::default()
    }
    .update_with_event(&context.db, &context.redis)
    .await?;
    Ok(())
}

fn sub_status_active(status: &stripe::SubscriptionStatus) -> bool {
    match status {
        stripe::SubscriptionStatus::Active | stripe::SubscriptionStatus::Trialing => true,
        stripe::SubscriptionStatus::Canceled
        | stripe::SubscriptionStatus::Incomplete
        | stripe::SubscriptionStatus::IncompleteExpired
        | stripe::SubscriptionStatus::PastDue
        | stripe::SubscriptionStatus::Paused
        | stripe::SubscriptionStatus::Unpaid => false,
    }
}

pub fn get_subscription_config(
    stripe_plans_config: &StripePlansConfig,
    license_type: elodin_types::api::LicenseType,
) -> Option<&StripePlanConfig> {
    match license_type {
        elodin_types::api::LicenseType::None | elodin_types::api::LicenseType::GodTier => None,
        elodin_types::api::LicenseType::NonCommercial => Some(&stripe_plans_config.non_commercial),
        elodin_types::api::LicenseType::Commercial => Some(&stripe_plans_config.commercial),
    }
}

impl Api {
    pub async fn get_stripe_subscription_status(
        &self,
        req: GetStripeSubscriptionStatusReq,
        CurrentUser { user, .. }: CurrentUser,
        txn: &DatabaseTransaction,
    ) -> Result<StripeSubscriptionStatus, Error> {
        let billing_account_id = Uuid::parse_str(&req.billing_account_id)?;

        self.get_subscription_status(billing_account_id, user.id, txn)
            .await
    }

    pub async fn get_subscription_status(
        &self,
        billing_account_id: Uuid,
        user_id: Uuid,
        txn: &impl ConnectionTrait,
    ) -> Result<StripeSubscriptionStatus, Error> {
        let tracing_debug_span = tracing::debug_span!(
            "get_subscription_status",
            %user_id,
            %billing_account_id,
        );

        async {
            tracing::debug!("get_subscription_status - start");

            let Some(billing_account) = billing_account::Entity::find_by_id(billing_account_id)
                .filter(billing_account::Column::OwnerUserId.eq(user_id))
                .one(txn)
                .await?
            else {
                tracing::error!("get_subscription_status - billing_account is missing");
                return Err(Error::NotFound);
            };

            let Ok(customer_id) = CustomerId::from_str(&billing_account.customer_id) else {
                tracing::error!("get_subscription_status - customer_id has wrong format");
                return Err(Error::NotFound);
            };

            let customer = Customer::retrieve(&self.stripe, &customer_id, &[]).await?;

            let portal_session = stripe::BillingPortalSession::create(
                &self.stripe,
                stripe::CreateBillingPortalSession {
                    return_url: Some(&self.base_url),
                    ..stripe::CreateBillingPortalSession::new(customer.id.clone())
                },
            )
            .await?;

            let default_subscription_status = StripeSubscriptionStatus {
                portal_url: portal_session.url,
                subscription_end: 0,
                trial_start: None,
                trial_end: None,
                monte_carlo_credit: 0,
            };

            let Some(sub_config) = get_subscription_config(
                &self.stripe_plans_config,
                billing_account.seat_license_type.into(),
            ) else {
                tracing::debug!("get_subscription_status - subscription_config is None");
                return Ok(default_subscription_status);
            };

            let Ok(price_id) = PriceId::from_str(&sub_config.subscription_price) else {
                tracing::error!("get_subscription_status - price_id has wrong format");
                return Err(Error::NotFound);
            };

            let list_subs_params = stripe::ListSubscriptions {
                customer: Some(customer_id),
                price: Some(price_id),
                ..stripe::ListSubscriptions::new()
            };

            let subs = stripe::Subscription::list(&self.stripe, &list_subs_params).await?;
            let Some(sub) = subs.data.first() else {
                // Subscription should always exists (even if it's ended)
                tracing::error!(
                    "get_subscription_status - subscription is missing from list response"
                );
                return Ok(default_subscription_status);
            };

            Ok(StripeSubscriptionStatus {
                subscription_end: sub.current_period_end,
                trial_start: sub.trial_start,
                trial_end: sub.trial_end,
                monte_carlo_credit: sub_config.monte_carlo_credit,
                ..default_subscription_status
            })
        }
        .instrument(tracing_debug_span)
        .await
    }
}

#[derive(FromQueryResult, Debug, Serialize, Deserialize)]
pub struct MonteCarloBilling {
    pub id: Uuid,
    pub name: String,
    pub user_id: Uuid,
    pub started: Option<DateTime<Utc>>,
    pub customer_id: String,
    pub usage_subscription_id: Option<String>,
    pub runtime_sum: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StripePlanSimple {
    pub id: String,
    pub meter: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StripeSubscriptionSimple {
    pub id: String,
    pub plan: StripePlanSimple,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StripeMeterSimple {
    pub id: String,
    pub event_name: String,
}
