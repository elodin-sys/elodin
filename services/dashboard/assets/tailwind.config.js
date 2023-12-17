// See the Tailwind configuration guide for advanced usage
// https://tailwindcss.com/docs/configuration

const plugin = require("tailwindcss/plugin")
const fs = require("fs")
const path = require("path")

module.exports = {
  content: [
    "./js/**/*.js",
    "../lib/elodin_dashboard_web.ex",
    "../lib/elodin_dashboard_web/**/*.*ex"
  ],
  theme: {
    extend: {
      colors: {
        "hyper-blue": "#1E43E2",
        "hyper-blue-dim": "#1736b5",
        blue: "#1E43E2",
        "light-blue": "#8EA1F1",
        tan: "#F3E5C5",
        orange: "#EF5800",
        green: "#00CA71",
        "dark-matte": "#131212",
        "code": "#282C34",
        "primative-colors-brand-hyper-red": "rgba(238, 58, 67, 1)",
        "primative-colors-white-opacity-100": "rgba(255, 255, 255, 0.1)",
        "primative-colors-white-opacity-50": "rgba(255, 255, 255, 0.05)",
        "primative-colors-white-opacity-900": "rgba(255, 255, 255, 1)",
        "tokens-surface-primary": "rgba(31, 31, 31, 1)",
        "tokens-surface-secondary": "rgba(13, 13, 13, 1)",
        "surface-secondary": "#1F1F1F",
        "surface-secondary-opacity-500": "rgba(31, 31, 31, 0.9)",
        "black-secondary": "rgba(13, 13, 13, 1)",
        "white-opacity": {
          "300": "rgba(255, 255, 255, 0.3)",
          "200": "rgba(255, 255, 255, 0.2)"
        },
      },
      spacing: {
        'elo-xxs': "4px",
        'elo-m': "8px",
        'elo-md': "12px",
        'elo-lg': "16px",
        'elo-xl': "24px",
      },
      borderRadius: {
        'elo-xs': "4px",
        'elo-sm': "8px",
        'elo-md': "12px"
      },
      fontFamily: {
        'sans': ["IBM Plex Sans", "ui-sans-serif", "system-ui", "-apple-system", "linkMacSystemFont", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "Noto Sans", "sans-serif", "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"]
      }
    },
  },
  plugins: [
    require("@tailwindcss/forms"),
    // Allows prefixing tailwind classes with LiveView classes to add rules
    // only when LiveView classes are applied, for example:
    //
    //     <div class="phx-click-loading:animate-ping">.
    //
    plugin(({addVariant}) => addVariant("phx-no-feedback", [".phx-no-feedback&", ".phx-no-feedback &"])),
    plugin(({addVariant}) => addVariant("phx-click-loading", [".phx-click-loading&", ".phx-click-loading &"])),
    plugin(({addVariant}) => addVariant("phx-submit-loading", [".phx-submit-loading&", ".phx-submit-loading &"])),
    plugin(({addVariant}) => addVariant("phx-change-loading", [".phx-change-loading&", ".phx-change-loading &"])),

    // Embeds Heroicons (https://heroicons.com) into your app.css bundle
    // See your `CoreComponents.icon/1` for more information.
    //
    plugin(function({matchComponents, theme}) {
      let iconsDir = path.join(__dirname, "./vendor/heroicons/optimized")
      let values = {}
      let icons = [
        ["", "/24/outline"],
        ["-solid", "/24/solid"],
        ["-mini", "/20/solid"]
      ]
      icons.forEach(([suffix, dir]) => {
        fs.readdirSync(path.join(iconsDir, dir)).forEach(file => {
          let name = path.basename(file, ".svg") + suffix
          values[name] = {name, fullPath: path.join(iconsDir, dir, file)}
        })
      })
      matchComponents({
        "hero": ({name, fullPath}) => {
          let content = fs.readFileSync(fullPath).toString().replace(/\r?\n|\r/g, "")
          return {
            [`--hero-${name}`]: `url('data:image/svg+xml;utf8,${content}')`,
            "-webkit-mask": `var(--hero-${name})`,
            "mask": `var(--hero-${name})`,
            "mask-repeat": "no-repeat",
            "background-color": "currentColor",
            "vertical-align": "middle",
            "display": "inline-block",
            "width": theme("spacing.5"),
            "height": theme("spacing.5")
          }
        }
      }, {values})
    })
  ]
}
