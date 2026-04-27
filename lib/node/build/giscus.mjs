import { existsSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { ensureDirForFile, safeRead, upsertStatus } from "./helpers.mjs";

const GISCUS_CONFIG_REL = "assets/core/giscus.config.js";

const DEFAULT_GISCUS_CONFIG = {
	repo: "",
	repoId: "",
	category: "",
	categoryId: "",
	mapping: "pathname",
	strict: "0",
	reactionsEnabled: "1",
	emitMetadata: "0",
	inputPosition: "bottom",
	loading: "lazy",
	lang: "zh-CN",
};

// 从 site.config.json 读取 giscus 字段；缺失或类型错误时回落到默认值。
function readGiscusConfig(siteConfigPath) {
	const raw = safeRead(siteConfigPath).trim();
	if (!raw) {
		return { ...DEFAULT_GISCUS_CONFIG };
	}

	let parsed;
	try {
		parsed = JSON.parse(raw);
	} catch {
		return { ...DEFAULT_GISCUS_CONFIG };
	}

	const source = parsed && typeof parsed === "object" && parsed.giscus && typeof parsed.giscus === "object"
		? parsed.giscus
		: {};

	const pickString = (value, fallback) => {
		if (value === undefined || value === null) {
			return fallback;
		}
		return String(value).trim();
	};

	return {
		repo: pickString(source.repo, DEFAULT_GISCUS_CONFIG.repo),
		repoId: pickString(source.repoId, DEFAULT_GISCUS_CONFIG.repoId),
		category: pickString(source.category, DEFAULT_GISCUS_CONFIG.category),
		categoryId: pickString(source.categoryId, DEFAULT_GISCUS_CONFIG.categoryId),
		mapping: pickString(source.mapping, DEFAULT_GISCUS_CONFIG.mapping),
		strict: pickString(source.strict, DEFAULT_GISCUS_CONFIG.strict),
		reactionsEnabled: pickString(source.reactionsEnabled, DEFAULT_GISCUS_CONFIG.reactionsEnabled),
		emitMetadata: pickString(source.emitMetadata, DEFAULT_GISCUS_CONFIG.emitMetadata),
		inputPosition: pickString(source.inputPosition, DEFAULT_GISCUS_CONFIG.inputPosition),
		loading: pickString(source.loading, DEFAULT_GISCUS_CONFIG.loading),
		lang: pickString(source.lang, DEFAULT_GISCUS_CONFIG.lang),
	};
}

// 把 giscus 配置写成 ES module 形式的 JS 文件，供前端 import 使用。
function buildConfigJs(config) {
	const serialized = JSON.stringify(config, null, 2);
	return `export const GISCUS_CONFIG = ${serialized};\n`;
}

// 将 giscus 配置写入 staging，并和旧产物对比设置 statusMap 状态。
export function stageGiscusConfig(siteConfigPath, outputSiteDir, stagingSiteDir, statusMap) {
	const config = readGiscusConfig(siteConfigPath);
	const fileContent = buildConfigJs(config);
	const oldOutputPath = join(outputSiteDir, GISCUS_CONFIG_REL);
	const stagingPath = join(stagingSiteDir, GISCUS_CONFIG_REL);

	ensureDirForFile(stagingPath);
	const unchanged = existsSync(oldOutputPath) && safeRead(oldOutputPath) === fileContent;
	writeFileSync(stagingPath, fileContent, "utf8");
	upsertStatus(statusMap, GISCUS_CONFIG_REL, unchanged ? "unchanged" : "updated");
}
