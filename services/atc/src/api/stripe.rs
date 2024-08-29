use crate::{config::StripePlansConfig, error::Error};
use atc_entity::{billing_account, events::DbExt, user};
use axum::{
    async_trait,
    extract::{FromRequest, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use chrono::{DateTime, Utc};
use elodin_types::api::{GetStripeSubscriptionStatusReq, LicenseType, StripeSubscriptionStatus};
use migration::Expr;
use reqwest::Client;
use sea_orm::{
    ColumnTrait, ConnectionTrait, DatabaseConnection, DatabaseTransaction, EntityTrait,
    FromQueryResult, JoinType, QueryFilter, QuerySelect, Set, Unchanged, Value,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, str::FromStr};
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

pub fn get_subscription_config(
    stripe_plans_config: &StripePlansConfig,
    license_type: elodin_types::api::LicenseType,
) -> Option<(String, u64, u32)> {
    match license_type {
        elodin_types::api::LicenseType::None | elodin_types::api::LicenseType::GodTier => None,
        elodin_types::api::LicenseType::NonCommercial => Some((
            stripe_plans_config.non_commercial_price.to_string(),
            30 * 3600 * 24,
            60,
        )),
        elodin_types::api::LicenseType::Commercial => Some((
            stripe_plans_config.commercial_price.to_string(),
            5 * 3600 * 24,
            60000,
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

        let Some((price_id_str, _, monte_carlo_credit)) = get_subscription_config(
            &self.stripe_plans_config,
            billing_account.seat_license_type.into(),
        ) else {
            return Ok(StripeSubscriptionStatus {
                portal_url: portal_session.url,
                subscription_end: 0,
                trial_start: None,
                trial_end: None,
                monte_carlo_credit: 0,
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
            monte_carlo_credit,
        })
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

pub(crate) async fn sync_monte_carlo_billing(
    client: &Client,
    db_connection: &DatabaseConnection,
    stripe_secret_key: &String,
) -> Result<(), Error> {
    let stripe_base_url = "https://api.stripe.com/v1";

    tracing::debug!("checking latest completed monte-carlo runs not saved in the stripe");

    // NOTE: Query monte-carlo runs with an attached total runtime and billing information
    let mc_run_billings = atc_entity::MonteCarloRun::find()
        .select_only()
        .columns([
            atc_entity::mc_run::Column::Id,
            atc_entity::mc_run::Column::Name,
            atc_entity::mc_run::Column::UserId,
            atc_entity::mc_run::Column::Started,
        ])
        .column_as(
            atc_entity::billing_account::Column::UsageSubscriptionId,
            "usage_subscription_id",
        )
        .column_as(
            atc_entity::billing_account::Column::CustomerId,
            "customer_id",
        )
        .column_as(atc_entity::batches::Column::Runtime.sum(), "runtime_sum")
        .filter(atc_entity::mc_run::Column::Billed.eq(false))
        .filter(atc_entity::mc_run::Column::Status.eq(atc_entity::mc_run::Status::Done))
        .join_rev(
            JoinType::InnerJoin,
            atc_entity::Batches::belongs_to(atc_entity::MonteCarloRun)
                .from(atc_entity::batches::Column::RunId)
                .to(atc_entity::mc_run::Column::Id)
                .into(),
        )
        .join_rev(
            JoinType::LeftJoin,
            atc_entity::BillingAccount::belongs_to(atc_entity::MonteCarloRun)
                .from(atc_entity::billing_account::Column::OwnerUserId)
                .to(atc_entity::mc_run::Column::UserId)
                .into(),
        )
        .group_by(atc_entity::mc_run::Column::Id)
        .group_by(atc_entity::billing_account::Column::UsageSubscriptionId)
        .group_by(atc_entity::billing_account::Column::CustomerId)
        .into_model::<MonteCarloBilling>()
        .all(db_connection)
        .await?;

    let amount_of_mc_runs_to_bill = mc_run_billings.len();
    tracing::debug!(%amount_of_mc_runs_to_bill, "got information about latest completed monte-carlo runs");

    let mut billed_mc_run_ids = vec![];

    for mc_run_billing in mc_run_billings {
        let Some(usage_subscription_id) = mc_run_billing.usage_subscription_id else {
            tracing::error!(%mc_run_billing.user_id, "user is missing usage_subscription_id, skipping");
            continue;
        };

        let runtime_min_str =
            ((mc_run_billing.runtime_sum as f64 / 60.0).ceil() as u64).to_string();
        let timestamp_str = mc_run_billing
            .started
            .map(|t| (t.timestamp() as u64).to_string());
        let mc_run_id_str = mc_run_billing.id.to_string();

        let sub_url = format!("{stripe_base_url}/subscriptions/{usage_subscription_id}");
        let stripe_subscription = client
            .get(sub_url)
            .basic_auth(stripe_secret_key, Some(""))
            .send()
            .await?
            .json::<StripeSubscriptionSimple>()
            .await?;

        let meter_url = format!(
            "{stripe_base_url}/billing/meters/{}",
            stripe_subscription.plan.meter
        );
        let stripe_meter = client
            .get(meter_url)
            .basic_auth(stripe_secret_key, Some(""))
            .send()
            .await?
            .json::<StripeMeterSimple>()
            .await?;

        tracing::debug!(%stripe_subscription.id, %stripe_meter.event_name, %mc_run_id_str, "received subscription information from stripe");

        let meter_event_url = format!("{stripe_base_url}/billing/meter_events");

        let mut meter_event_params = HashMap::new();
        meter_event_params.insert("event_name", stripe_meter.event_name);
        meter_event_params.insert("payload[value]", runtime_min_str);
        meter_event_params.insert("payload[stripe_customer_id]", mc_run_billing.customer_id);
        meter_event_params.insert(
            "identifier",
            format!("{mc_run_id_str}/{}", mc_run_billing.name),
        );
        if let Some(timestamp_str) = timestamp_str {
            meter_event_params.insert("timestamp", timestamp_str);
        }

        let meter_event_res = client
            .post(meter_event_url)
            .basic_auth(stripe_secret_key, Some(""))
            .form(&meter_event_params)
            .send()
            .await?;

        let create_meter_event_response_status = meter_event_res.status();
        tracing::debug!(%create_meter_event_response_status, "received response from stripe");

        if !meter_event_res.status().is_success() {
            let create_meter_event_response = meter_event_res.text().await?;
            tracing::error!(%create_meter_event_response, "something went wrong during the creation of a meter_event");
        } else {
            billed_mc_run_ids.push(mc_run_billing.id);
        }
    }

    let amount_of_billed_mc_runs = billed_mc_run_ids.len();

    if amount_of_billed_mc_runs > 0 {
        tracing::debug!(%amount_of_billed_mc_runs, "updating billed monte-carlo runs in the database");

        let update_query = atc_entity::MonteCarloRun::update_many()
            .col_expr(
                atc_entity::mc_run::Column::Billed,
                Expr::value(Value::Bool(Some(true))),
            )
            .filter(atc_entity::mc_run::Column::Id.is_in(billed_mc_run_ids))
            .exec(db_connection)
            .await?;

        tracing::debug!(%update_query.rows_affected, "finished updating `monte_carlo_runs` table");
    }

    Ok(())
}
