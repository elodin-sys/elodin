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
      animation: {
        sliding: 'sliding 0.5s ease 1',
      },
      colors: {
        "hyper-blue": "#1E43E2",
        "hyper-blue-dim": "#1736b5",
        blue: "#1E43E2",
        "light-blue": "#8EA1F1",
        orange: {
          DEFAULT: "#EF5800",
          50: "#FFFBF0",
        },
        green: {
          DEFAULT: "#88DE9F",
          40: "rgb(30 32 28)",
          300: "#88DE9F",
        },
        "green-opacity": {
          "40": "rgba(136, 222, 159, 0.4)"
        },
        "dark-matte": "#131212",
        "red": {
          DEFAULT: "rgb(233, 75, 20)",
          40: "rgb(31, 21, 17)",
          30: "rgba(233, 75, 20, 0.3)",
          5: "rgba(233, 75, 20, 0.05)"
        },
        yellow: {
          DEFAULT: "rgba(254, 197, 4, 1)",
          400: "#FEC504",
        },
        "tan": "rgb(255, 215, 179)",
        "code": "#282C34",
        "primative-colors-brand-hyper-red": "rgba(238, 58, 67, 1)",
        "primative-colors-white-opacity-100": "rgba(255, 255, 255, 0.1)",
        "primative-colors-white-opacity-50": "rgba(255, 255, 255, 0.05)",
        "primative-colors-white-opacity-900": "rgba(255, 255, 255, 1)",
        "tokens-surface-primary": "rgba(31, 31, 31, 1)",
        "tokens-surface-secondary": "rgba(13, 13, 13, 1)",
        "crema": {
          DEFAULT: "rgba(255, 251, 240, 1)",
          60:  "rgba(255, 251, 240, 0.6)",
        },
        "black-primary": "rgba(23, 22, 21, 1)",
        "black-secondary": "rgba(13, 13, 13, 1)",
        "black-header": "rgb(20, 20, 20)",
        "black-opacity": {
          "600": "rgba(0, 0, 0, 0.6)",
        },
        "white-opacity": {
          "30": "rgba(255, 255, 255, 0.3)",
          "20": "rgba(255, 255, 255, 0.2)",
          "10": "rgba(255, 255, 255, 0.1)"
        },
        "sep-black": "#202020",
        "primary-onyx": {
          "9": "rgba(46, 45, 44, 1)",
        },
        "primary-smoke": "rgba(13, 13, 13, 1)",
      },
      spacing: {
        'elo-xxs': "4px",
        'elo-m': "8px",
        'elo-md': "12px",
        'elo-lg': "16px",
        'elo-xl': "24px",
      },
      height: {
        '128': '32rem',
      },
      borderRadius: {
        'elo-xxs': "2px",
        'elo-xs': "4px",
        'elo-sm': "8px",
        'elo-md': "12px"
      },
      gridTemplateColumns: {
        '10': 'repeat(10, minmax(0, 1fr))',
        '16': 'repeat(16, minmax(0, 1fr))',
        '18': 'repeat(18, minmax(0, 1fr))',
        '20': 'repeat(20, minmax(0, 1fr))',
      },
      fontSize: {
        sm: '13px'
      },
      fontFamily: {
        'sans': ["IBM Plex Sans", "ui-sans-serif", "system-ui", "-apple-system", "linkMacSystemFont", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "Noto Sans", "sans-serif", "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"],
        'mono': ["IBM Plex Mono", "ui-monospace", "mono"]
      },
      screens: {
        "elo-grid-xl": "1400px",
      },
      letterSpacing: {
        "elo-mono-medium": "1.04px",
        "elo-mono-small": "0.48px"
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
