use zbus::fdo::ManagedObjects;
use zbus::{Connection, proxy, zvariant::OwnedObjectPath};

#[proxy(
    interface = "net.connman.iwd.Station",
    default_service = "net.connman.iwd",
    default_path = "/net/connman/iwd/0"
)]
pub trait Station {
    /// Scan for wireless networks
    async fn scan(&self) -> zbus::Result<()>;

    /// Get list of available networks with their signal strength
    fn get_ordered_networks(&self) -> zbus::Result<Vec<(OwnedObjectPath, i16)>>;

    /// Get currently connected network if any
    #[zbus(property)]
    fn connected_network(&self) -> zbus::Result<OwnedObjectPath>;

    /// Disconnect from current network
    async fn disconnect(&self) -> zbus::Result<()>;
}

#[proxy(
    interface = "net.connman.iwd.Device",
    default_service = "net.connman.iwd",
    default_path = "/net/connman/iwd/0"
)]
pub trait Device {
    /// Get device name
    #[zbus(property)]
    fn name(&self) -> zbus::Result<String>;

    /// Get device MAC address
    #[zbus(property)]
    fn address(&self) -> zbus::Result<String>;

    /// Check if device is powered on
    #[zbus(property)]
    fn powered(&self) -> zbus::Result<bool>;

    /// Power on/off the device
    #[zbus(property)]
    fn set_powered(&self, powered: bool) -> zbus::Result<()>;
}

#[proxy(
    interface = "net.connman.iwd.Network",
    default_service = "net.connman.iwd",
    default_path = "/net/connman/iwd/0"
)]
pub trait Network {
    /// Get network name (SSID)
    #[zbus(property)]
    fn name(&self) -> zbus::Result<String>;

    /// Get network type (e.g., "psk" for WPA/WPA2)
    #[zbus(property)]
    fn type_(&self) -> zbus::Result<String>;

    /// Connect to this network
    async fn connect(&self) -> zbus::Result<()>;
}

#[proxy(
    interface = "net.connman.iwd.KnownNetwork",
    default_service = "net.connman.iwd",
    default_path = "/net/connman/iwd"
)]
pub trait KnownNetwork {
    /// Get network name (SSID)
    #[zbus(property)]
    fn name(&self) -> zbus::Result<String>;

    /// Get network type
    #[zbus(property)]
    fn type_(&self) -> zbus::Result<String>;

    /// Forget this network and its configuration
    async fn forget(&self) -> zbus::Result<()>;
}

#[proxy(
    interface = "net.connman.iwd.AgentManager",
    default_service = "net.connman.iwd",
    default_path = "/net/connman/iwd"
)]
pub trait AgentManager {
    async fn register_agent(&self, path: OwnedObjectPath) -> zbus::Result<()>;
    async fn unregister_agent(&self, path: OwnedObjectPath) -> zbus::Result<()>;
}

pub struct Iwd {
    objects: ManagedObjects,
}

pub struct Agent {
    password: String,
}

impl Agent {
    pub fn new(password: String) -> Self {
        Self { password }
    }
}

#[zbus::interface(name = "net.connman.iwd.Agent")]
impl Agent {
    async fn request_passphrase(&self, _network: OwnedObjectPath) -> String {
        self.password.clone()
    }
}

impl Iwd {
    pub async fn new(connection: &Connection) -> zbus::Result<Self> {
        let object_manager =
            zbus::fdo::ObjectManagerProxy::new(connection, "net.connman.iwd", "/").await?;
        let objects = object_manager.get_managed_objects().await?;
        Ok(Self { objects })
    }

    pub async fn station(&self, connection: &Connection) -> Option<StationProxy> {
        let (path, _) = self
            .objects
            .iter()
            .find(|(_, i)| i.contains_key("net.connman.iwd.Station"))?;
        let builder = StationProxy::builder(connection).path(path).ok()?;
        builder.build().await.ok()
    }
}

pub trait StationExt {
    async fn networks(&self, connection: &Connection) -> zbus::Result<Vec<NetworkProxy>>;
}

impl StationExt for StationProxy<'_> {
    async fn networks(&self, conn: &Connection) -> zbus::Result<Vec<NetworkProxy>> {
        let mut networks = vec![];
        for (path, _) in self.get_ordered_networks().await? {
            networks.push(NetworkProxy::builder(conn).path(path)?.build().await?);
        }
        Ok(networks)
    }
}
