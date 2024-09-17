## Playwright Automated Testing for app.elodin.dev
https://playwright.dev/

*note*: Before choosing Javascript Playwright, I investigated
- using Python, and discovered the API and capability are far behind the Javascript version, so much that you can't test our site.
- using a Rust-based Webdriver, such as [fantoccini](https://github.com/jonhoo/fantoccini) or [rust-headless-chrome](https://github.com/rust-headless-chrome/rust-headless-chrome)
- ultimately, the Javascript Playwright API is the most feature-rich and well-documented, and successfully tested the basic aspects of our web application.

The intention of our use of Playwright is simple & minimal sanity checking of web functionality, used sparingly in critical places.    
We are not using Playwright for extensive testing, or for testing the full range of functionality of the web application.  


### Install
use [fnm](https://github.com/Schniz/fnm) to install nodejs in a rust & fish friendly manner
```sh
curl -fsSL https://fnm.vercel.app/install | bash
fnm install # installs NodeJS according the local .node-version
npm install # installs Playwright according to the local package.json
npx playwright install # installs the Playwright browsers according to playwright.config.js
```

### Run the tests headlessly
This will run the tests in the `tests` directory, with 3 retries on failures, on all browsers configured in the playwright config, and output the results to the console, and then serve the results in browser for analysis.    
```sh
npx playwright test --retries=3
```

### Run tests in interactive mode
```sh
npx playwright test --ui
```

### To Generate a New Test
```sh
npx playwright codegen app.elodin.dev
```

### Modify / Customize Tests:
*note*: The generated tests will actually need a lot of work after generation to actually pass and robustly:
https://playwright.dev/docs/writing-tests

