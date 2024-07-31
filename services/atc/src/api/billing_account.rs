use atc_entity::{billing_account, events::DbExt, user};
use elodin_types::api::{BillingAccount, CreateBillingAccountReq};
use sea_orm::{ColumnTrait, DatabaseTransaction, EntityTrait, QueryFilter, Set};
use std::time::SystemTime;
use stripe::{CreateCustomer, Customer, Scheduled};
use uuid::Uuid;

use crate::error::Error;

use super::{stripe::get_price_id_and_trial, Api, CurrentUser};

impl Api {
    pub async fn create_billing_account(
        &self,
        req: CreateBillingAccountReq,
        CurrentUser { user, .. }: CurrentUser,
        txn: &DatabaseTransaction,
    ) -> Result<BillingAccount, Error> {
        let billing_account_id = Uuid::now_v7();

        let existing_accounts = billing_account::Entity::find()
            .filter(billing_account::Column::OwnerUserId.eq(user.id))
            .all(txn)
            .await?;
        let customer = Customer::create(
            &self.stripe,
            CreateCustomer {
                name: Some(&user.name),
                email: Some(&user.email),
                metadata: Some(std::collections::HashMap::from([
                    ("user-id".to_string(), user.id.to_string()),
                    (
                        "billing-account-id".to_string(),
                        billing_account_id.to_string(),
                    ),
                ])),

                ..Default::default()
            },
        )
        .await?;
        billing_account::ActiveModel {
            id: Set(billing_account_id),
            name: Set(req.name.clone()),
            customer_id: Set(customer.id.to_string()),
            owner_user_id: Set(user.id),
            seat_subscription_id: Set(None),
            usage_subscription_id: Set(None),
            monte_carlo_active: Set(false),
            seat_count: Set(0),
            seat_license_type: Set((!existing_accounts.is_empty())
                .then(|| req.trial_license_type().into())
                .unwrap_or(atc_entity::user::LicenseType::None)),
        }
        .insert_with_event(&self.db, &self.redis)
        .await?;
        if existing_accounts.is_empty() {
            let (price_id, trial_length) =
                get_price_id_and_trial(&self.stripe_plans_config, req.trial_license_type())?;
            let trial_end = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + trial_length;
            let trial_end = trial_end as i64;
            stripe::Subscription::create(
                    &self.stripe,
                    stripe::CreateSubscription {
                        trial_end: Some(Scheduled::at(trial_end)),
                        items: Some(vec![stripe::CreateSubscriptionItems {
                            price: Some(price_id),
                            ..Default::default()
                        }]),
                        trial_settings: Some(stripe::CreateSubscriptionTrialSettings {
                            end_behavior: stripe::CreateSubscriptionTrialSettingsEndBehavior {
                                missing_payment_method: stripe::CreateSubscriptionTrialSettingsEndBehaviorMissingPaymentMethod::Pause
                            }
                        }),
                        metadata: Some(std::collections::HashMap::from([(
                            "sub_type".to_string(),
                            "seat".to_string(),
                        ),
                        (
                            "billing_account_id".to_string(),
                            billing_account_id.to_string(),
                        )
                        ])),
                        ..stripe::CreateSubscription::new(customer.id.clone())
                    },
                )
                .await?;
            user::ActiveModel {
                id: Set(user.id),
                billing_account_id: Set(Some(billing_account_id)),
                ..Default::default()
            }
            .update_with_event(&self.db, &self.redis)
            .await?;
        }

        stripe::Subscription::create(
            &self.stripe,
            stripe::CreateSubscription {
                items: Some(vec![stripe::CreateSubscriptionItems {
                    price: Some(self.stripe_plans_config.monte_carlo_price.to_string()),
                    ..Default::default()
                }]),
                metadata: Some(std::collections::HashMap::from([
                    ("sub_type".to_string(), "monte-carlo".to_string()),
                    (
                        "billing_account_id".to_string(),
                        billing_account_id.to_string(),
                    ),
                ])),
                ..stripe::CreateSubscription::new(customer.id.clone())
            },
        )
        .await?;

        Ok(BillingAccount {
            id: billing_account_id.as_bytes().to_vec(),
            name: req.name,
            customer_id: customer.id.to_string(),
        })
    }
}
