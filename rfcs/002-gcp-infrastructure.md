
# [Cloud Infrastructure on GCP](https://linear.app/elodin/project/initial-cloud-setup-afbb83bf4903)


## Goals

- It should be possible to move the Paracosm cluster to another cloud infrastructure (GCP > AWS/Azure/etc) without any issues.
- Resources should be easily managed.
- We should be able to scale in either direction without issues.


## Approach

### Terraform

While there are alternatives to `Terraform` (like [Pulumi](https://www.pulumi.com/)), we already have a lot of experience with this setup and not enough problems to search for an alternative at this time. With that being said first iteration will only include provisioning from a local machine, and in the following project [I plan to look into options](https://linear.app/elodin/issue/ELO-26) that will allow us to manage all resources remotely without local setup.

At this point, only a few things are really necessary for our infrastructure:
- Cluster resources can be found in `terraform/gcp/cluster`
    - Region, machine type, and number of nodes can be managed through variables
- GCP Project resources (e.g. artifact repos) can be found in `terraform/gcp/project`
- Domain records can be found in `terraform/cloudflare`

This setup will allow us to change and move parts of the project without affecting everything.

#### Cluster resources

Currently, we plan to optimize our spending through [spot instances](https://cloud.google.com/kubernetes-engine/docs/concepts/spot-vms) and [autoscaling](https://cloud.google.com/kubernetes-engine/docs/concepts/cluster-autoscaler).

Regarding `spot instances`, our first setup will use only them, but depending on the results we probably would want to have multiple node pools with some of them using non-spot instances for pods that require a bit more stable server. 

Unfortunately, it's not possible to scale the amount of CPUs in a single node "on the fly", so at first it would be better to use more machines with a smaller amount of CPUs. If you look at [pricing for n1-standard here](https://cloud.google.com/compute/all-pricing#general_purpose) you'll see that 1 CPU cost is the same as whatever you choose `n1-standard-1` or `n1-standard-96`.

### Kubernetes

The primary use of this setup will be to demonstrate simple Monte Carlo runs on Paracosm using a Web Application. So based on these requirements we will need to deploy the following items:

- `Ingress` with configured load balancer and domain for public accessible services
- `Deployment` for Web App / Frontend 
- `Deployment` for ATC (Air Traffic Controller) / Backend API
- `Deployment` with Paracosm Simulator pods (considering using [Kata containers](https://katacontainers.io/))
- SQL Database to store user information (settings, projects, etc.). At this moment we will be okay with our own simple `Deployment`, but managing our own database deployment is often not worth it and better option would be to use an external solution (CockroachDB currently was proposed as an option)

If we use multiple node pools we will need to set up taints and tolerations for said deployments.
