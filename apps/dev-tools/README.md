
# Dev Tools CLI

## Auth0

```sh
# Get Auth0 token from https://manage.auth0.com/dashboard/us/dev-i2ytsp68gngieek3/apis/management/explorer
export ELODIN_AUTH0_TOKEN="..."

# Get all users and export to auth0_users.csv
cargo run -- auth0 get-users --csv auth0_users
# Get all users that have email containing `playwright`
cargo run -- auth0 get-users --email '*playwright*'

# Remove user with 'auth0|66e638163bd291855593277c' id
cargo run -- auth0 delete-user-by-id --user-id 'auth0|66e638163bd291855593277c'
```


## Stripe


```sh
# Get Stripe token from 1Pass (Infrastrucure/Stripe)
export ELODIN_STRIPE_TOKEN="..."

# Get all customers and export to stripe_users.csv
cargo run -- stripe get-users --csv stripe_users
# Get all customers that have email `andrei@elodin.systems`
cargo run -- stripe get-users --email andrei@elodin.systems
# Get all customers that have `andrei` in their email
cargo run -- stripe get-users --email andrei --approximate

# Delete user by id
cargo run -- stripe delete-user --user-id cus_QPFeDObIvTP9YE
```


## Postgres


```sh
# Port-forward postgres database from cluster (replace `CLUSTER_NAME`)
kubectl port-forward service/postgres 5432:5432 -n elodin-app-CLUSTER_NAME

# Get all users and export to pg_users.csv
cargo run -- postgres-db get-users --csv pg_users
# Get all users that have email containing `playwright`
cargo run -- postgres-db get-users --email '%playwright%'

# Delete user by id
cargo run -- postgres-db delete-user --user-id 01924f72-79ce-7312-9568-cbfbc01b37a3
```

## Others

```sh
# Tokens for auth0/stripe are required, and port-forward to postgres
# Search all services for user with `elodin.systems` substring in their email
cargo run -- get-users --email 'elodin.systems'

# Delete user from all associated services using postgres user-id
cargo run -- delete-user --user-id 01925825-c5a4-7260-be57-6013e919b5d4
```

## Production Changes

To apply changes and run queries in production environment following steps are necessary:
- Set production token for `auth0` and `stripe`
- Connect to a production cluster (the same port-forward command will work)
- Use `--prod` flag in the CLI
