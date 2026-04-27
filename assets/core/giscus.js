import { GISCUS_CONFIG } from "./giscus.config.js";

const GISCUS_CLIENT_URL = "https://giscus.app/client.js";
const GISCUS_ORIGIN = "https://giscus.app";
const CONTAINER_SELECTOR = ".giscus";
const IFRAME_SELECTOR = "iframe.giscus-frame";

// Carbon 4 档主题 -> Giscus 主题的细分映射。
const THEME_MAP = {
	"white": "light_high_contrast",
	"gray-10": "light",
	"gray-90": "dark_dimmed",
	"gray-100": "dark",
};

const DEFAULT_GISCUS_THEME = "preferred_color_scheme";

const isConfigComplete = (config) => {
	if (!config || typeof config !== "object") {
		return false;
	}

	const required = ["repo", "repoId", "category", "categoryId"];
	return required.every((key) => typeof config[key] === "string" && config[key].trim().length > 0);
};

const resolveGiscusTheme = () => {
	const dataTheme = document.documentElement.getAttribute("data-theme");
	return THEME_MAP[dataTheme] || DEFAULT_GISCUS_THEME;
};

const buildGiscusScript = (config, theme) => {
	const script = document.createElement("script");
	script.src = GISCUS_CLIENT_URL;
	script.async = true;
	script.crossOrigin = "anonymous";
	script.setAttribute("data-repo", config.repo);
	script.setAttribute("data-repo-id", config.repoId);
	script.setAttribute("data-category", config.category);
	script.setAttribute("data-category-id", config.categoryId);
	script.setAttribute("data-mapping", config.mapping || "pathname");
	script.setAttribute("data-strict", config.strict || "0");
	script.setAttribute("data-reactions-enabled", config.reactionsEnabled || "1");
	script.setAttribute("data-emit-metadata", config.emitMetadata || "0");
	script.setAttribute("data-input-position", config.inputPosition || "bottom");
	script.setAttribute("data-theme", theme);
	script.setAttribute("data-lang", config.lang || "zh-CN");
	script.setAttribute("data-loading", config.loading || "lazy");
	return script;
};

const syncThemeToIframe = (theme) => {
	const iframe = document.querySelector(IFRAME_SELECTOR);
	if (!iframe || !iframe.contentWindow) {
		return;
	}

	iframe.contentWindow.postMessage(
		{ giscus: { setConfig: { theme } } },
		GISCUS_ORIGIN,
	);
};

const observeThemeChanges = () => {
	const observer = new MutationObserver((mutations) => {
		for (const mutation of mutations) {
			if (mutation.type === "attributes" && mutation.attributeName === "data-theme") {
				syncThemeToIframe(resolveGiscusTheme());
				return;
			}
		}
	});

	observer.observe(document.documentElement, {
		attributes: true,
		attributeFilter: ["data-theme"],
	});
};

const initGiscus = () => {
	const container = document.querySelector(CONTAINER_SELECTOR);
	if (!container) {
		return;
	}

	if (!isConfigComplete(GISCUS_CONFIG)) {
		console.warn("[giscus] missing repoId or categoryId in site.config.json; comments disabled.");
		return;
	}

	const initialTheme = resolveGiscusTheme();
	const script = buildGiscusScript(GISCUS_CONFIG, initialTheme);
	container.appendChild(script);
	observeThemeChanges();
};

if (document.readyState === "loading") {
	document.addEventListener("DOMContentLoaded", initGiscus, { once: true });
} else {
	initGiscus();
}
