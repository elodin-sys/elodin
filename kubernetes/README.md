
# Kubernetes

## Applying changes to the project

```sh
# Login into the development cluster
gcloud container clusters get-credentials "elodin-dev-gke" --region "us-central1" --project "elodin-dev" 

# Run following commands from the root of the repo

kubectl kustomize kubernetes/overlays/dev > out.yaml
BUILDKITE_COMMIT=$(git rev-parse HEAD) envsubst < out.yaml > out-with-envs.yaml

kubectl apply -f out-with-envs.yaml

kubectl get all -n elodin-app-dev

# NOTE: Use following command before `terraform destroy` if you intend to fully remove all resources
kubectl delete -f out-with-envs.yaml
```

## Production deployment

```sh
# Login into the production cluster
gcloud container clusters get-credentials "elodin-prod-gke" --region "us-central1" --project "elodin-prod"

# Run following commands from the root of the repo

export RELEASE_IMAGE_TAG="0.2.0"

just re-tag-images-main $RELEASE_IMAGE_TAG

kubectl kustomize kubernetes/overlays/prod > out.yaml
envsubst < out.yaml > out-with-envs.yaml

kubectl apply -f out-with-envs.yaml

kubectl get all -n elodin-app-prod
```

## Project structure

Kubernetes folder contains everything necessary to configure a cluster.

`elodin-app` folders contains all resources that would be placed in namespaces with the respective names
