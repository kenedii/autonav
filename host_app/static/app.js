let cars = [];
let selectedCarId = null;
let pollMs = 2000;
let pollTimer = null;
let experiments = [];
let uploadedDeployModelPath = "";
let uploadedJetsonOptPath = "";
let uploadedRockchipOptPath = "";

let statusChart = null;
let logTimer = null;
let logCursor = {};
let logStore = {};

let videoSocket = null;
let videoCtx = null;

const BACKBONE_MAP = {
    "resnet34": "ResNet-34",
    "resnet101": "ResNet-101",
    "resnet152": "ResNet-152"
};

function byId(id) {
    return document.getElementById(id);
}

function selectedCar() {
    return cars.find((c) => c.id === selectedCarId);
}

function getStatusDotClass(status) {
    if (status === "online") return "dot-green";
    if (status === "stopped") return "dot-red";
    return "dot-gray";
}

function readBackbone(prefix) {
    const value = byId(`${prefix}-backbone`).value;
    if (value !== "custom") return value;
    const custom = byId(`${prefix}-backbone-custom`).value.trim();
    return custom || "resnet34";
}

function activeClientTab() {
    const active = document.querySelector(".client-tab-btn.active");
    return active ? active.dataset.clientTab : "overview";
}

function setupTopTabs() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const tab = btn.dataset.tab;
            document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach((pane) => pane.classList.remove("active"));
            btn.classList.add("active");
            byId(tab).classList.add("active");
        });
    });
}

function setupClientTabs() {
    document.querySelectorAll(".client-tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const tab = btn.dataset.clientTab;
            document.querySelectorAll(".client-tab-btn").forEach((b) => b.classList.remove("active"));
            document.querySelectorAll(".client-tab-pane").forEach((pane) => pane.classList.remove("active"));
            btn.classList.add("active");
            byId(`client-tab-${tab}`).classList.add("active");

            if (tab !== "video") {
                stopVideoStream();
            }
            if (tab === "logs") {
                startLogPolling();
            } else {
                stopLogPolling();
            }
        });
    });
}

function setupOptimizeTabs() {
    document.querySelectorAll(".opt-tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const tab = btn.dataset.optTab;
            document.querySelectorAll(".opt-tab-btn").forEach((b) => b.classList.remove("active"));
            document.querySelectorAll(".opt-pane").forEach((p) => p.classList.remove("active"));
            btn.classList.add("active");
            byId(`opt-tab-${tab}`).classList.add("active");
        });
    });
}

function setMessage(elId, text) {
    byId(elId).textContent = text;
}

function setJson(elId, data) {
    byId(elId).textContent = JSON.stringify(data, null, 2);
}

async function fetchHostInfo() {
    try {
        const res = await fetch("/api/host-info");
        const data = await res.json();
        setMessage("host-ip-info", `Server IP: ${data.ip}:${data.port}`);
    } catch (e) {
        setMessage("host-ip-info", "Server IP: unavailable");
    }
}

async function fetchExperiments() {
    try {
        const res = await fetch("/api/experiments");
        const data = await res.json();
        experiments = data.experiments || [];
        populateExperimentSelectors();
        refreshExperimentDescription();
    } catch (e) {
        console.error("Failed to fetch experiments", e);
    }
}

function populateExperimentSelectors() {
    const targets = ["deploy-exp", "trt-exp", "rknn-exp"];
    targets.forEach((id) => {
        const select = byId(id);
        const prev = select.value;
        select.innerHTML = "";
        experiments.forEach((exp) => {
            const opt = document.createElement("option");
            opt.value = String(exp.experiment);
            opt.textContent = `Experiment ${exp.experiment}`;
            select.appendChild(opt);
        });
        if (prev) select.value = prev;
    });
}

function refreshExperimentDescription() {
    const exp = Number(byId("deploy-exp").value || 1);
    const found = experiments.find((x) => x.experiment === exp);
    if (!found) {
        setMessage("deploy-exp-description", "No experiment metadata available.");
        return;
    }
    setMessage("deploy-exp-description", `Sensors used: ${found.description}`);
}

async function fetchCars() {
    try {
        const res = await fetch("/api/cars");
        cars = await res.json();
        renderFleetList();
        updateStats();
        updateCharts();
        if (selectedCarId) {
            updateDetailPanel();
        }
    } catch (e) {
        console.error("Failed to fetch cars", e);
    }
}

function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    fetchCars();
    pollTimer = setInterval(fetchCars, pollMs);
}

function updateStats() {
    byId("stat-total").textContent = String(cars.length);
    byId("stat-active").textContent = String(cars.filter((c) => c.status === "online").length);
}

function renderFleetList() {
    const list = byId("car-list");
    list.innerHTML = "";

    cars.forEach((car) => {
        const li = document.createElement("li");
        li.className = `car-item ${selectedCarId === car.id ? "selected" : ""}`;
        li.innerHTML = `
            <div class="car-item-row">
                <div><span class="status-dot ${getStatusDotClass(car.status)}"></span><strong>${car.name}</strong></div>
                <small>${car.status || "unknown"}</small>
            </div>
            <div class="car-ip">${car.ip}:${car.port || 8000}</div>
        `;
        li.addEventListener("click", () => selectCar(car.id));
        list.appendChild(li);
    });
}

async function selectCar(id) {
    if (selectedCarId === id) return;
    stopVideoStream();
    stopLogPolling();

    selectedCarId = id;
    uploadedDeployModelPath = "";
    uploadedJetsonOptPath = "";
    uploadedRockchipOptPath = "";
    setMessage("uploaded-model-label", "No file uploaded");
    setMessage("trt-upload-state", "No model uploaded");
    setMessage("rknn-upload-state", "No model uploaded");
    setMessage("deploy-model-path", "");
    setJson("deploy-result", { status: "waiting" });
    setJson("optimize-result", { status: "waiting" });
    disableDownload();

    renderFleetList();

    byId("empty-state").classList.add("hidden");
    byId("car-detail-content").classList.remove("hidden");

    try {
        await fetch(`/api/cars/${id}/status`);
    } catch (e) {
        console.error(e);
    }

    await updateDetailPanel();
}

async function updateDetailPanel() {
    const car = selectedCar();
    if (!car) return;

    byId("detail-name").textContent = car.name;
    byId("detail-ip").textContent = `${car.ip}:${car.port || 8000}`;
    byId("detail-status").textContent = (car.status || "unknown").toUpperCase();

    const geo = car.geo || {};
    byId("detail-geo").textContent = geo.city ? `${geo.city}, ${geo.country}` : "Unknown / Local";

    const details = car.details || {};
    byId("detail-activity").textContent = details.running ? (details.paused ? "Running (Paused)" : "Running") : "Stopped";
    byId("detail-fps").textContent = String(details.state?.fps || 0);
    byId("spec-inference").textContent = details.state?.specs?.inference || "Unknown";

    const specs = details.state?.specs || {};
    const modelSummary = {
        backend: car.details?.config?.control_model_type || "unknown",
        model_path: car.details?.config?.control_model || "not set",
        architecture: car.details?.config?.architecture || specs.resnet_version || "unknown",
        experiment: car.details?.config?.experiment || "unknown",
        sensors: car.details?.config?.experiment_sensors || [],
        front_camera: car.details?.config?.front_camera || "not set",
        rear_camera: car.details?.config?.rear_camera || "not set"
    };
    setJson("loaded-model-info", modelSummary);

    populateCameraOptions(specs.cameras || []);
    await refreshPlatformBadge();

    if (activeClientTab() === "logs") startLogPolling();
}

function populateCameraOptions(cameras) {
    const select = byId("camera-select");
    select.innerHTML = "";

    if (!Array.isArray(cameras) || cameras.length === 0) {
        const opt = document.createElement("option");
        opt.value = "0";
        opt.textContent = "Camera 0";
        select.appendChild(opt);
        return;
    }

    cameras.forEach((cam, index) => {
        const opt = document.createElement("option");
        const camIndex = cam.index !== undefined ? cam.index : index;
        opt.value = String(camIndex);
        opt.textContent = `Camera ${camIndex}: ${cam.type || "camera"}`;
        select.appendChild(opt);
    });
}

async function addTestClient() {
    await fetch("/api/test-client", { method: "POST" });
    await fetchCars();
}

async function deleteCar() {
    if (!selectedCarId) return;
    if (!window.confirm("Remove this car from fleet?")) return;
    await fetch(`/api/cars/${selectedCarId}`, { method: "DELETE" });
    selectedCarId = null;
    byId("car-detail-content").classList.add("hidden");
    byId("empty-state").classList.remove("hidden");
    await fetchCars();
}

async function controlAction(action) {
    if (!selectedCarId) return;
    await fetch(`/api/cars/${selectedCarId}/${action}`, { method: "POST" });
    setJson("deploy-result", { status: `sent ${action}` });
    await fetchCars();
}

async function updateSettings() {
    if (!selectedCarId) return;
    const throttle_mode = byId("tune-mode").value;
    const fixed_throttle_value = Number(byId("tune-throttle").value || 0.22);
    await fetch(`/api/cars/${selectedCarId}/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ throttle_mode, fixed_throttle_value })
    });
    setJson("deploy-result", { status: "settings updated", throttle_mode, fixed_throttle_value });
}

async function uploadDeployModel() {
    if (!selectedCarId) return;
    const file = byId("deploy-model-file").files[0];
    if (!file) {
        setMessage("uploaded-model-label", "Select a model file first");
        return;
    }

    const fd = new FormData();
    fd.append("file", file);
    fd.append("category", "control");

    const res = await fetch(`/api/cars/${selectedCarId}/models/upload`, {
        method: "POST",
        body: fd
    });
    const data = await res.json();
    if (!res.ok) {
        setJson("deploy-result", data);
        return;
    }
    uploadedDeployModelPath = data.path;
    byId("deploy-model-path").value = data.path;
    setMessage("uploaded-model-label", `Uploaded: ${data.filename}`);
}

function buildCamerasFromForm() {
    const front = {
        role: "front",
        type: byId("front-cam-source").value,
        index: Number(byId("front-cam-index").value || 0),
        width: 640,
        height: 480,
        fps: 15
    };

    const useRear = byId("rear-enabled").checked;
    const cameras = [front];

    if (useRear) {
        cameras.push({
            role: "rear",
            type: byId("rear-cam-source").value,
            index: Number(byId("rear-cam-index").value || 1),
            width: 640,
            height: 480,
            fps: 15
        });
    }

    return cameras;
}

async function deployControlModel() {
    if (!selectedCarId) return;
    const experiment = Number(byId("deploy-exp").value || 1);
    const expInfo = experiments.find((e) => e.experiment === experiment);
    const architecture = readBackbone("deploy");
    const modelPath = byId("deploy-model-path").value.trim();

    if (!modelPath) {
        setJson("deploy-result", { error: "Upload model before deploying." });
        return;
    }

    const cameras = buildCamerasFromForm();
    const config = {
        device: "cuda",
        architecture,
        experiment,
        experiment_features: expInfo ? expInfo.features : [],
        experiment_sensors: expInfo ? expInfo.sensors : [],
        front_camera: {
            driver: byId("front-cam-source").value,
            index: Number(byId("front-cam-index").value || 0)
        },
        rear_camera: byId("rear-enabled").checked ? {
            driver: byId("rear-cam-source").value,
            index: Number(byId("rear-cam-index").value || 1)
        } : null,
        use_depth: byId("use-depth").checked,
        use_ir: byId("use-ir").checked,
        cameras,
        control_model_type: byId("deploy-model-backend").value,
        control_model: modelPath,
        detection_model: "yolov8n.pt",
        throttle_mode: byId("tune-mode").value,
        fixed_throttle_value: Number(byId("tune-throttle").value || 0.22),
        action_loop: ["control", "api"]
    };

    const res = await fetch(`/api/cars/${selectedCarId}/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config })
    });
    const data = await res.json();
    setJson("deploy-result", data);

    if (res.ok) {
        setJson("loaded-model-info", {
            deployed: true,
            backend: config.control_model_type,
            architecture: config.architecture,
            experiment: config.experiment,
            sensors: config.experiment_sensors,
            model_path: config.control_model,
            front_camera: config.front_camera,
            rear_camera: config.rear_camera,
            use_depth: config.use_depth,
            use_ir: config.use_ir
        });
    }
}

async function refreshPlatformBadge() {
    if (!selectedCarId) return;
    try {
        const res = await fetch(`/api/cars/${selectedCarId}/platform`);
        const data = await res.json();
        setMessage("platform-badge", `Platform: ${data.platform || "unknown"}`);

        if ((data.platform || "").toLowerCase() === "jetson") {
            byId("opt-tab-jetson").classList.add("active");
            byId("opt-tab-rockchip").classList.remove("active");
            document.querySelectorAll(".opt-tab-btn").forEach((b) => b.classList.remove("active"));
            document.querySelector('.opt-tab-btn[data-opt-tab="jetson"]').classList.add("active");
        } else if ((data.platform || "").toLowerCase() === "rockchip") {
            byId("opt-tab-rockchip").classList.add("active");
            byId("opt-tab-jetson").classList.remove("active");
            document.querySelectorAll(".opt-tab-btn").forEach((b) => b.classList.remove("active"));
            document.querySelector('.opt-tab-btn[data-opt-tab="rockchip"]').classList.add("active");
        }
    } catch (e) {
        setMessage("platform-badge", "Platform: unknown");
    }
}

async function uploadJetsonOptimizeModel() {
    if (!selectedCarId) return;
    const file = byId("trt-model-file").files[0];
    if (!file) {
        setMessage("trt-upload-state", "Select a model file first");
        return;
    }

    const fd = new FormData();
    fd.append("file", file);
    fd.append("category", "optimize");

    const res = await fetch(`/api/cars/${selectedCarId}/models/upload`, { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
        setJson("optimize-result", data);
        return;
    }
    uploadedJetsonOptPath = data.path;
    setMessage("trt-upload-state", `Uploaded: ${data.filename}`);
}

async function runTensorRtOptimize() {
    if (!selectedCarId) return;
    if (!uploadedJetsonOptPath) {
        setJson("optimize-result", { error: "Upload a model to the car before optimization." });
        return;
    }

    setJson("optimize-result", { status: "running TensorRT optimization" });

    const payload = {
        experiment: Number(byId("trt-exp").value || 1),
        architecture: byId("trt-arch").value,
        model_path: uploadedJetsonOptPath
    };

    const res = await fetch(`/api/cars/${selectedCarId}/models/optimize/tensorrt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
    const data = await res.json();
    setJson("optimize-result", data);

    if (res.ok && data.artifact_id) {
        enableDownload(data.artifact_id, data.filename || "best_model_trt.pth");
    }
}

async function uploadRockchipOptimizeModel() {
    const file = byId("rknn-model-file").files[0];
    if (!file) {
        setMessage("rknn-upload-state", "Select a model file first");
        return;
    }

    const fd = new FormData();
    fd.append("file", file);
    fd.append("category", "rockchip");

    const res = await fetch("/api/host/models/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
        setJson("optimize-result", data);
        return;
    }

    uploadedRockchipOptPath = data.path;
    setMessage("rknn-upload-state", `Uploaded: ${data.filename}`);
}

async function runRknnOptimize() {
    if (!uploadedRockchipOptPath) {
        setJson("optimize-result", { error: "Upload a model to the host before RKNN optimization." });
        return;
    }

    setJson("optimize-result", { status: "running RKNN optimization on x86 host" });

    const payload = {
        experiment: Number(byId("rknn-exp").value || 1),
        architecture: byId("rknn-arch").value,
        model_path: uploadedRockchipOptPath,
        timeout: 3600
    };

    const res = await fetch("/api/optimize/rknn/x86", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await res.json();
    setJson("optimize-result", data);
    if (res.ok && data.artifact_id) {
        enableDownload(data.artifact_id, data.filename || "best_model.rknn");
    }
}

function enableDownload(artifactId, filename) {
    const link = byId("download-optimized-link");
    link.href = `/api/artifacts/${artifactId}/download`;
    link.download = filename;
    link.classList.remove("disabled-link");
}

function disableDownload() {
    const link = byId("download-optimized-link");
    link.href = "#";
    link.download = "";
    link.classList.add("disabled-link");
}

function startVideoStream() {
    if (!selectedCarId) return;
    stopVideoStream();

    const canvas = byId("video-canvas");
    if (!videoCtx) videoCtx = canvas.getContext("2d");

    const encodedId = encodeURIComponent(selectedCarId);
    const cameraIndex = Number(byId("camera-select").value || 0);
    const fps = Number(byId("fps-select").value || 5);
    const wsProto = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${wsProto}://${window.location.host}/ws/video/${encodedId}?camera_index=${cameraIndex}&fps=${fps}`;

    videoSocket = new WebSocket(wsUrl);
    videoSocket.binaryType = "arraybuffer";
    setMessage("video-status", "Connecting...");

    videoSocket.onopen = () => setMessage("video-status", "Streaming");
    videoSocket.onclose = () => setMessage("video-status", "Stopped");
    videoSocket.onerror = () => setMessage("video-status", "Error");

    videoSocket.onmessage = (ev) => {
        if (!(ev.data instanceof ArrayBuffer)) return;
        const blob = new Blob([ev.data], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
            videoCtx.drawImage(img, 0, 0, canvas.width, canvas.height);
            URL.revokeObjectURL(url);
        };
        img.src = url;
    };
}

function stopVideoStream() {
    if (videoSocket) {
        videoSocket.close();
        videoSocket = null;
    }
    setMessage("video-status", "Stopped");
}

async function fetchLogs() {
    if (!selectedCarId) return;
    const since = logCursor[selectedCarId] || 0;
    try {
        const res = await fetch(`/api/cars/${selectedCarId}/logs?since=${since}`);
        const data = await res.json();
        const newLogs = data.logs || [];
        if (!logStore[selectedCarId]) logStore[selectedCarId] = [];
        logStore[selectedCarId] = logStore[selectedCarId].concat(newLogs);

        if (newLogs.length > 0) {
            logCursor[selectedCarId] = newLogs[newLogs.length - 1].timestamp;
        }
        renderLogs();
    } catch (e) {
        console.error("log fetch failed", e);
    }
}

function renderLogs() {
    const out = byId("log-output");
    const logs = logStore[selectedCarId] || [];
    if (logs.length === 0) {
        out.textContent = "No logs yet.";
        return;
    }

    out.textContent = logs
        .map((e) => {
            const ts = new Date((e.timestamp || 0) * 1000).toISOString();
            return `[${ts}] [${(e.level || "INFO").toUpperCase()}] ${e.message || ""}`;
        })
        .join("\n");
    out.scrollTop = out.scrollHeight;
}

function startLogPolling() {
    stopLogPolling();
    fetchLogs();
    logTimer = setInterval(fetchLogs, 1200);
}

function stopLogPolling() {
    if (logTimer) {
        clearInterval(logTimer);
        logTimer = null;
    }
}

async function clearLogs() {
    if (!selectedCarId) return;
    await fetch(`/api/cars/${selectedCarId}/logs`, { method: "DELETE" });
    logStore[selectedCarId] = [];
    logCursor[selectedCarId] = 0;
    renderLogs();
}

function saveLogs() {
    if (!selectedCarId) return;
    const text = (logStore[selectedCarId] || [])
        .map((e) => {
            const ts = new Date((e.timestamp || 0) * 1000).toISOString();
            return `[${ts}] [${(e.level || "INFO").toUpperCase()}] ${e.message || ""}`;
        })
        .join("\n");

    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `car_${selectedCarId}_logs.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

function updateCharts() {
    const canvas = byId("statusChart");
    if (!canvas) return;

    const online = cars.filter((c) => c.status === "online").length;
    const stopped = cars.filter((c) => c.status === "stopped").length;
    const offline = cars.length - online - stopped;

    if (!statusChart) {
        statusChart = new Chart(canvas.getContext("2d"), {
            type: "doughnut",
            data: {
                labels: ["Running", "Stopped", "Offline"],
                datasets: [{
                    data: [online, stopped, offline],
                    backgroundColor: ["#3ecf8e", "#f4bf4f", "#f25f5c"]
                }]
            },
            options: {
                plugins: {
                    legend: {
                        labels: { color: "#e9f4f8" }
                    }
                }
            }
        });
        return;
    }

    statusChart.data.datasets[0].data = [online, stopped, offline];
    statusChart.update();
}

function saveGlobalSettings() {
    const next = Number(byId("setting-poll").value || 2000);
    pollMs = Math.max(500, next);
    startPolling();
}

function wireUiActions() {
    byId("add-test-client-btn").addEventListener("click", addTestClient);
    byId("remove-car-btn").addEventListener("click", deleteCar);

    document.querySelectorAll("button[data-control]").forEach((btn) => {
        btn.addEventListener("click", () => controlAction(btn.dataset.control));
    });

    byId("save-tuning-btn").addEventListener("click", updateSettings);
    byId("upload-model-btn").addEventListener("click", uploadDeployModel);
    byId("deploy-model-btn").addEventListener("click", deployControlModel);

    byId("deploy-exp").addEventListener("change", refreshExperimentDescription);
    byId("deploy-backbone").addEventListener("change", () => {
        const isCustom = byId("deploy-backbone").value === "custom";
        byId("deploy-custom-wrap").classList.toggle("hidden", !isCustom);
    });

    byId("trt-upload-btn").addEventListener("click", uploadJetsonOptimizeModel);
    byId("run-trt-btn").addEventListener("click", runTensorRtOptimize);
    byId("rknn-upload-btn").addEventListener("click", uploadRockchipOptimizeModel);
    byId("run-rknn-btn").addEventListener("click", runRknnOptimize);

    byId("video-start-btn").addEventListener("click", startVideoStream);
    byId("video-stop-btn").addEventListener("click", stopVideoStream);

    byId("clear-logs-btn").addEventListener("click", clearLogs);
    byId("save-logs-btn").addEventListener("click", saveLogs);

    byId("save-global-settings-btn").addEventListener("click", saveGlobalSettings);
}

async function init() {
    setupTopTabs();
    setupClientTabs();
    setupOptimizeTabs();
    wireUiActions();

    await fetchHostInfo();
    await fetchExperiments();
    startPolling();
}

window.addEventListener("load", init);
