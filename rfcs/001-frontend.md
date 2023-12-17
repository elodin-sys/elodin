# Choosing a Frontend Framework
Sascha Wise - 2023-11-15

**TLDR** - We will use Phoenix, it is a good combination of stable, easy to use, fast, and feature rich. 

We need a front-end web framework for Elodin. There are an almost infinite number of options in the frontend world, of wildly varying quality. We are looking for a framework that solves a few things:

1. Ease of development - whatever we use needs to be easy and fast to work with
2. Fast - users should not notice any slowness from the frontend framework
3. Maintainable - Many frontend frameworks are easy to get started with, but become unusable as the code-base grows in complexity

There are a few anti-goals as well:

1. Ceremony - Lots of frontend systems require a great deal of boilerplate and BS ceremony - coughs NextJS - let's avoid that
2. Package management woes - NPM is famous for breaking all the time
3. Popular - In the short-term popularity shouldn't necessarily determine the tools we use. We are likely going to be managing the front end as a small team, so hiring should be less of a concern.

## Options

**React** (and its many derivatives)

React is super popular, but we've all dealt with its problems. It is filled with ceremony, has the worst package manager, and can be quite slow. Overall for this team, it is best avoided. 

**Svelte, Vue, et al**

It might not be super fair to compare all of these frameworks, but broadly they are all React competitors. They do not have React's main advantage, its popularity, and they still fall victim to many of the JS framework problems. They often win out in ease of development and/or speed.

**Dioxus**

Dioxus is a new Rust web framework that allows 3 forms of rendering SSR, LiveView (streaming HTML over WebSockets), and WASM-SPA. It looks pretty fantastic, and like it could be the future of Rust web development. A code base that is written 100% in Rust is very appealing. The biggest cons are that it appears to be fairly early, with potential API changes inbound, and the most supported flow requires downloading a WASM bundle. 

**Phoenix**

Phoenix is an Elixir web framework similar in concept to Rails, Django, or ASP.net Core. It has one major feature over those, [LiveView](https://hexdocs.pm/phoenix_live_view/Phoenix.LiveView.html). LiveView is sort of a hybrid between an SPA and an old-school SSR MVC web framework. Phoenix will set up a WebSocket for each page load, when an event happens on the client side, those changes will be sent back to the web server. You get the reactivity of a SPA and the ease of development of a standard web framework. The biggest downside for us is adding another programming language. Elixir, thankfully, is quite a nice language. Fully functional, has a good package manager, and is built on the wildly stable Erlang VM.

## Decision
I see Dioxus and Phoenix as the most viable options. One of my biggest concerns with JS-based web frameworks is the wide amount of ceremony that is required to use them. State is difficult to manage; requests to backends must be proxied, often through multiple layers; and dependencies are painful to manage. Plus JS is just not that fun to use. Diouxs has the benefit of using Rust, a language we are already deep into. Phoenix has the advantage of a larger community, more usage, and just being more stable. To me, those points win out over the single-language goals.

I will take a moment to discuss the hire-ability issue, random frontend guys off the street will not know Elixir. I view that as an advantage. In my experience, teams are the fastest at developing new front-end features when they are empowered to make the required backend changes as well. When there is a disconnect between frontend and backend teams, things start to get slow. I think that is one of the primary reasons why Rails was so effective. There was no separation, just features. I want us to build a team with no separation, if you need to add a new feature it gets added to the API and the front end. To do that we need to hire polyglots, people who can code Rust, Elixir, HTML, JS, whatever. 
