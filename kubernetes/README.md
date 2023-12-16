
# Kubernetes

## Applying changes to the project

```sh

kubectl kustomize overlays/dev > out.yaml

kubectl apply -f out.yaml

kubectl get all -n elodin-app-dev

# NOTE: it's necessary for `terraform destroy`, since you're not managing `Network endpoint group` in there
# NOTE: also disks created for PVC are not removed unless related k8s resource is deleted
kubectl delete -f out.yaml
```

## Project structure

Kubernetes folder contains everything necessary to configure a cluster.

`elodin-app`/`elodin-vms` folders contains all resources that would be placed in namespaces with the respective names

## Extra Notes

### Example of deployment manifest configuration for `gvisor`

```yml
spec:
  template:
    spec:
      runtimeClassName: gvisor
```
