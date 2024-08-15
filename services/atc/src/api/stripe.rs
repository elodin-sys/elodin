use crate::{config::StripePlansConfig, error::Error};
use atc_entity::{billing_account, events::DbExt, user};
use axum::{
    async_trait,
    extract::{FromRequest, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use elodin_types::api::{GetStripeSubscriptionStatusReq, LicenseType, StripeSubscriptionStatus};
use sea_orm::{
    ColumnTrait, ConnectionTrait, DatabaseTransaction, EntityTrait, QueryFilter, Set, Unchanged,
};
use std::str::FromStr;
use stripe::{Customer, CustomerId, Event, EventObject, EventType, PriceId};
use tracing::warn;
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
                warn!("no billing account id in subscription metadata");
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
                            id if id == context.stripe_plans_config.commercial_price => {
                                LicenseType::Commercial
                            }
                            id if id == context.stripe_plans_config.non_commercial_price => {
                                LicenseType::NonCommercial
                            }
                            id if id == context.stripe_plans_config.monte_carlo_price => {
                                warn!("found monte carlo price in seat subscription");
                                continue;
                            }
                            price_id => {
                                warn!(?price_id, "unknown price id in seat subscription");
                                continue;
                            }
                        };
                        if license_type == LicenseType::None {
                            license_type = new_license_type;
                        } else if license_type != new_license_type {
                            warn!(
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
                        item.price.as_ref().map(|price| price.id.as_str())
                            == Some(&context.stripe_plans_config.monte_carlo_price)
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
                    warn!(?sub_type, "unknown subscription type");
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

pub fn get_price_id_and_trial(
    stripe_plans_config: &StripePlansConfig,
    license_type: elodin_types::api::LicenseType,
) -> Option<(String, u64)> {
    match license_type {
        elodin_types::api::LicenseType::None | elodin_types::api::LicenseType::GodTier => None,
        elodin_types::api::LicenseType::NonCommercial => Some((
            stripe_plans_config.non_commercial_price.to_string(),
            30 * 3600 * 24,
        )),
        elodin_types::api::LicenseType::Commercial => Some((
            stripe_plans_config.commercial_price.to_string(),
            5 * 3600 * 24,
        )),
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
        let Some(billing_account) = billing_account::Entity::find_by_id(billing_account_id)
            .filter(billing_account::Column::OwnerUserId.eq(user_id))
            .one(txn)
            .await?
        else {
            return Err(Error::NotFound);
        };

        let Ok(customer_id) = CustomerId::from_str(&billing_account.customer_id) else {
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

        let Some((price_id_str, _)) = get_price_id_and_trial(
            &self.stripe_plans_config,
            billing_account.seat_license_type.into(),
        ) else {
            return Ok(StripeSubscriptionStatus {
                portal_url: portal_session.url,
                subscription_end: 0,
                trial_start: None,
                trial_end: None,
            });
        };

        let Ok(price_id) = PriceId::from_str(&price_id_str) else {
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
            return Err(Error::NotFound);
        };

        Ok(StripeSubscriptionStatus {
            portal_url: portal_session.url,
            subscription_end: sub.current_period_end,
            trial_start: sub.trial_start,
            trial_end: sub.trial_end,
        })
    }
}
