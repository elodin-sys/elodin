use anyhow::anyhow;
use fuzzy_matcher::{FuzzyMatcher, skim::SkimMatcherV2};
use iwd::{Agent, AgentManagerProxy, StationExt};
use nu_ansi_term::{Color, Style};
use promkit::preset::confirm::Confirm;
use promkit::preset::listbox::Listbox;
use promkit::preset::password::Password;
use promkit::preset::query_selector::QuerySelector;
use promkit::preset::readline::Readline;
use std::{fmt::Display, fs, path::Path, sync::OnceLock, time::Duration};
use tokio::{fs::File, io::AsyncWriteExt, process::Command};
use zbus::zvariant::OwnedObjectPath;
mod iwd;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!();
    print_header("Welcome to ℵ Aleph!", Color::Purple);
    println!();
    let should_connect = Confirm::new("Do you want to connect to WiFi?")
        .prompt()?
        .run()?
        == "y";
    if should_connect {
        connect_to_wifi().await?;
    }

    let should_create_user = Confirm::new("Do you want to create a user account?")
        .prompt()?
        .run()?
        == "y";
    if should_create_user {
        let username = create_user().await?;
        add_user_public_key(&username).await?;
    }

    Ok(())
}

fn divider_line(color: Color) -> &'static String {
    match color {
        Color::Green => {
            static DIVIDER_LINE: OnceLock<String> = OnceLock::new();
            DIVIDER_LINE.get_or_init(|| Color::Green.paint("▌").to_string())
        }
        Color::Yellow => {
            static DIVIDER_LINE: OnceLock<String> = OnceLock::new();
            DIVIDER_LINE.get_or_init(|| Color::Yellow.paint("▌").to_string())
        }

        Color::Purple => {
            static DIVIDER_LINE: OnceLock<String> = OnceLock::new();
            DIVIDER_LINE.get_or_init(|| Color::Purple.paint("▌").to_string())
        }
        Color::Blue => {
            static DIVIDER_LINE: OnceLock<String> = OnceLock::new();
            DIVIDER_LINE.get_or_init(|| Color::Blue.paint("▌").to_string())
        }
        Color::Red => {
            static DIVIDER_LINE: OnceLock<String> = OnceLock::new();
            DIVIDER_LINE.get_or_init(|| Color::Red.paint("▌").to_string())
        }

        _ => unimplemented!("unsupported divider color"),
    }
}

fn print_header(text: impl Display, color: Color) {
    println!(
        "{}{}",
        divider_line(color),
        Style::new()
            .bold()
            .on(color)
            .fg(Color::Black)
            .paint(format!(" {text} "))
    );
}

async fn connect_to_wifi() -> anyhow::Result<()> {
    loop {
        let connection = zbus::Connection::system().await?;
        let iwd = iwd::Iwd::new(&connection).await?;
        let station = iwd
            .station(&connection)
            .await
            .ok_or_else(|| anyhow!("no station found"))?;
        station.scan().await?;
        let networks = station.networks(&connection).await?;
        let mut network_names = vec![];
        for network in &networks {
            network_names.push(network.name().await?);
        }

        let mut query = QuerySelector::new(&network_names, |text, items| {
            let matcher = SkimMatcherV2::default();
            items
                .iter()
                .filter(|n| matcher.fuzzy_match(n, text).is_some())
                .cloned()
                .collect::<Vec<_>>()
        })
        .title("Please select a WiFi network to connect to")
        .listbox_lines(5)
        .prompt()?;
        let selected_network = query.run()?;
        let network_index = network_names
            .iter()
            .position(|name| name == &selected_network)
            .unwrap();
        let network = &networks[network_index];

        // Check if network requires a password
        let network_type = network.type_().await?;
        let connection_result = if network_type == "psk" {
            let title = format!("Enter the password for {}", selected_network);
            let mut password_prompt = Password::default().title(&title).prompt()?;
            let password = password_prompt.run()?;

            let agent = Agent::new(password);
            let agent_path = format!("/aleph/agent/{}", fastrand::u64(..));
            let agent_path = OwnedObjectPath::try_from(agent_path)?;
            let agent_manager = AgentManagerProxy::new(&connection).await?;
            agent_manager.register_agent(agent_path.clone()).await?;
            connection
                .object_server()
                .at(agent_path.clone(), agent)
                .await?;
            let res = network.connect().await;
            agent_manager.unregister_agent(agent_path.clone()).await?;
            res
        } else {
            network.connect().await
        };
        if connection_result.is_ok() {
            print_header(format!("Connected to {selected_network}"), Color::Green);
            break;
        } else {
            print_header("Connection Error", Color::Red);
        }
    }

    Ok(())
}

async fn create_user() -> anyhow::Result<String> {
    print_header("Create User Account", Color::Blue);

    let mut username_prompt = Readline::default().title("Enter username").prompt()?;
    let username = username_prompt.run()?;

    let mut password = None;
    loop {
        let mut password_prompt = Password::default()
            .title("Enter password for the new account")
            .prompt()?;
        let pass = password.insert(password_prompt.run()?);
        if pass.is_empty() {
            password = None;
            break;
        }

        let mut confirm_password_prompt = Password::default().title("Confirm password").prompt()?;
        let confirm_password = confirm_password_prompt.run()?;

        if pass == &confirm_password {
            break;
        }
        print_header("Passwords do not match", Color::Red);
    }

    let result = Command::new("useradd")
        .args([
            "-m",
            "-G",
            "wheel",
            "-s",
            "/run/current-system/sw/bin/bash",
            "-U",
            &username,
        ])
        .status()
        .await?;

    if !result.success() {
        print_header("Failed to create user", Color::Red);
        return Err(anyhow!("useradd command failed"));
    }

    if let Some(password) = password {
        let mut passwd_process = Command::new("chpasswd")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .spawn()?;

        if let Some(stdin) = &mut passwd_process.stdin {
            use tokio::io::AsyncWriteExt;
            stdin
                .write_all(format!("{}:{}", username, password).as_bytes())
                .await?;
            stdin.flush().await?;
        }

        let passwd_result = passwd_process.wait_with_output().await?;

        if passwd_result.status.success() {
            print_header(
                format!("User '{}' created successfully", username),
                Color::Green,
            );
        } else {
            print_header("Failed to set password", Color::Red);
            return Err(anyhow!("chpasswd command failed"));
        }
    }

    Ok(username)
}

async fn wait_for_internet() -> anyhow::Result<()> {
    print_header("Waiting for Internet Connection ...", Color::Blue);
    for _ in 0..256 {
        if reqwest::get("https://github.com").await.is_ok() {
            return Ok(());
        };
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    print_header("Failed to connect to internet", Color::Red);
    Err(anyhow!("Failed to connect to internet"))
}

async fn add_user_public_key(username: &str) -> anyhow::Result<()> {
    print_header("SSH Public Key Setup", Color::Blue);

    let options = vec!["Download from GitHub", "Enter manually", "Skip"];
    let mut listbox = Listbox::new(&options)
        .title("How would you like to add your SSH public key?")
        .prompt()?;

    let selection = listbox.run()?;

    match selection.as_str() {
        "Download from GitHub" => {
            wait_for_internet().await?;

            let mut github_username_prompt = Readline::default()
                .title("Enter your GitHub username")
                .prompt()?;
            let github_username = github_username_prompt.run()?;

            print_header("Downloading public keys from GitHub...", Color::Blue);
            println!();
            for _ in 0..32 {
                let url = format!("https://github.com/{}.keys", github_username);
                let Ok(response) = reqwest::get(&url).await else {
                    print_header("Failed to fetch keys from GitHub. Retrying ...", Color::Red);
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    continue;
                };

                if !response.status().is_success() {
                    print_header("Failed to fetch keys from GitHub. Retrying ...", Color::Red);
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    continue;
                }

                let public_keys = response.text().await?;

                if public_keys.trim().is_empty() {
                    print_header("No public keys found on GitHub", Color::Yellow);
                    return Ok(());
                }

                save_public_keys(username, &public_keys).await?;
                print_header("Public keys from GitHub added successfully", Color::Green);
                break;
            }
        }
        "Enter manually" => {
            let mut public_key_prompt = Readline::default()
                .title("Paste your public key (ssh-rsa or ssh-ed25519 format)")
                .prompt()?;
            let public_key = public_key_prompt.run()?;

            save_public_keys(username, &public_key).await?;
            print_header("Public key added successfully", Color::Green);
        }
        "Skip" => {
            print_header("Skipping SSH key setup", Color::Yellow);
        }
        _ => unreachable!(),
    }

    Ok(())
}

async fn save_public_keys(username: &str, public_keys: &str) -> anyhow::Result<()> {
    let home_dir = format!("/home/{}", username);
    let ssh_dir = format!("{}/.ssh", home_dir);

    if !Path::new(&ssh_dir).exists() {
        fs::create_dir_all(&ssh_dir)?;
        Command::new("chmod")
            .args(["700", &ssh_dir])
            .status()
            .await?;

        let res = Command::new("chown")
            .args(["-R", &format!("{}:{}", username, username), &ssh_dir])
            .status()
            .await?;
        if !res.success() {
            return Err(anyhow!("failed to run chown"));
        }
    }

    let authorized_keys_path = format!("{}/.ssh/authorized_keys", home_dir);
    let mut file = File::create(&authorized_keys_path).await?;

    for key_line in public_keys.lines() {
        if key_line.trim().is_empty() {
            continue;
        }
        file.write_all(key_line.trim().as_bytes()).await?;
        file.write_all(b"\n").await?;
    }

    Command::new("chmod")
        .args(["600", &authorized_keys_path])
        .status()
        .await?;

    let res = Command::new("chown")
        .args([&format!("{}:{}", username, username), &authorized_keys_path])
        .status()
        .await?;
    if !res.success() {
        return Err(anyhow!("failed to run chown"));
    }

    Ok(())
}
