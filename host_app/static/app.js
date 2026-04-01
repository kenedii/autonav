// State
let cars = [];
let selectedCarId = null;
let pollInterval = null;
let logPollInterval = null;
let pollIntervalMs = 2000;
let statusChart = null;
let videoSocket = null;
let videoSocketGeneration = 0;
let videoObjectUrl = null;
let logCursorByCarId = {};

function getSelectedCar() {
    if (!selectedCarId) return null;
    return cars.find(function (car) { return car.id === selectedCarId; }) || null;
}

function encodeCarId(carId) {
    return encodeURIComponent(carId || "");
}

function getHostWebSocketBase(path) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return protocol + "//" + window.location.host + path;
}

function setInlineStatus(message, kind) {
    const el = document.getElementById("operator-status");
    if (!el) return;
    el.textContent = message || "";
    el.className = "inline-status";
    if (kind) {
        el.classList.add(kind);
    }
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function setBadge(id, value, kind) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = value;
    el.className = "status-badge";
    if (kind) {
        el.classList.add(kind);
    }
}

function formatBool(value) {
    return value ? "Yes" : "No";
}

function formatDistanceMeters(value) {
    if (value === null || value === undefined || value === "") return "N/A";
    const num = Number(value);
    if (!isFinite(num)) return "N/A";
    return num.toFixed(2) + " m";
}

function formatVector(vec) {
    if (!vec || !vec.length) return "N/A";
    return "X:" + Number(vec[0]).toFixed(2) + " Y:" + Number(vec[1]).toFixed(2) + " Z:" + Number(vec[2]).toFixed(2);
}

const SENSOR_ROLE_META = [
    {
        key: "primary_rgb",
        title: "CAM0 Primary RGB",
        subtitle: "Forward control, AprilTags, and preview.",
    },
    {
        key: "sidecar_depth_imu",
        title: "RealSense Sidecar Depth/IMU",
        subtitle: "Depth stop and IMU context only.",
    },
    {
        key: "rear_preview",
        title: "CAM1 Rear Preview",
        subtitle: "Reverse-only scaffolding.",
    },
];

function sensorStatusText(sensor) {
    if (!sensor) return "unconfigured";
    if (sensor.status) return String(sensor.status);
    if (sensor.healthy === false) return "degraded";
    if (sensor.enabled === false) return "disabled";
    return "available";
}

function sensorStatusKind(sensor) {
    const status = sensorStatusText(sensor).toLowerCase();
    if (status.includes("available") || status.includes("healthy") || status === "ok") {
        return "ok";
    }
    if (status.includes("configured") || status.includes("standby") || status.includes("disabled") || status.includes("unconfigured")) {
        return "neutral";
    }
    if (status.includes("warn") || status.includes("degraded")) {
        return "warning";
    }
    return "danger";
}

function cameraSummary(camera) {
    if (!camera) return "unconfigured";
    const role = camera.role ? camera.role : "camera";
    const type = camera.type ? String(camera.type).toUpperCase() : "UNKNOWN";
    const size = camera.width && camera.height ? camera.width + "x" + camera.height : "n/a";
    const fps = camera.fps ? camera.fps + " fps" : "n/a";
    return role + " | " + type + " | " + size + " | " + fps;
}

function formatSensorDetail(sensor) {
    if (!sensor) return [];
    const lines = [];
    if (sensor.source) lines.push("source " + sensor.source);
    if (sensor.sensor_id !== undefined && sensor.sensor_id !== null) lines.push("sensor " + sensor.sensor_id);
    if (sensor.flip_method !== undefined && sensor.flip_method !== null) lines.push("flip " + sensor.flip_method);
    if (sensor.depth) lines.push("depth " + sensor.depth);
    if (sensor.imu) lines.push("imu " + sensor.imu);
    if (sensor.healthy !== undefined) lines.push("healthy " + formatBool(sensor.healthy));
    if (sensor.enabled !== undefined) lines.push("enabled " + formatBool(sensor.enabled));
    return lines;
}

function renderSensorRig(sensorRig) {
    const panel = document.getElementById("sensor-rig-panel");
    if (!panel) return;
    panel.innerHTML = "";

    const rig = sensorRig || {};
    const forwardRole = rig.forward_preview_role || "primary_rgb";
    const forwardLabel = forwardRole === "primary_rgb" ? "CAM0" : forwardRole;
    setBadge("forward-preview-role-badge", forwardLabel, forwardRole === "primary_rgb" ? "ok" : "neutral");

    SENSOR_ROLE_META.forEach(function (meta) {
        const sensor = rig[meta.key] || {};
        const card = document.createElement("article");
        card.className = "sensor-card";

        const header = document.createElement("div");
        header.className = "sensor-card-header";

        const titleWrap = document.createElement("div");
        const title = document.createElement("h4");
        title.textContent = meta.title;
        const subtitle = document.createElement("p");
        subtitle.className = "sensor-card-subtitle";
        subtitle.textContent = meta.subtitle;
        titleWrap.appendChild(title);
        titleWrap.appendChild(subtitle);

        const badge = document.createElement("span");
        badge.className = "status-badge " + sensorStatusKind(sensor);
        badge.textContent = sensorStatusText(sensor);

        header.appendChild(titleWrap);
        header.appendChild(badge);

        const summary = document.createElement("p");
        summary.className = "sensor-card-summary";
        summary.textContent = cameraSummary(sensor);

        const metaList = document.createElement("div");
        metaList.className = "sensor-meta";
        formatSensorDetail(sensor).forEach(function (line) {
            const row = document.createElement("div");
            row.className = "sensor-meta-row";
            row.textContent = line;
            metaList.appendChild(row);
        });
        if (!metaList.children.length) {
            const row = document.createElement("div");
            row.className = "sensor-meta-row";
            row.textContent = "no sensor details";
            metaList.appendChild(row);
        }

        card.appendChild(header);
        card.appendChild(summary);
        card.appendChild(metaList);
        panel.appendChild(card);
    });
}

function getCameraByRole(cameras, role) {
    if (!Array.isArray(cameras)) return null;
    const byRole = cameras.find(function (camera) {
        return camera && camera.role === role;
    });
    if (byRole) return byRole;
    if (role === "primary_rgb") {
        return cameras[0] || null;
    }
    return null;
}

function populateCameraSection(prefix, camera, defaults) {
    const cfg = camera || defaults || {};
    const enabledEl = document.getElementById("cfg-" + prefix + "-enabled");
    const typeEl = document.getElementById("cfg-" + prefix + "-type");
    const sensorIdEl = document.getElementById("cfg-" + prefix + "-sensor-id");
    const indexEl = document.getElementById("cfg-" + prefix + "-index");
    const widthEl = document.getElementById("cfg-" + prefix + "-w");
    const heightEl = document.getElementById("cfg-" + prefix + "-h");
    const fpsEl = document.getElementById("cfg-" + prefix + "-fps");
    const flipEl = document.getElementById("cfg-" + prefix + "-flip");

    if (enabledEl) enabledEl.checked = cfg.enabled !== false;
    if (typeEl) typeEl.value = cfg.type || defaults.type || "csi";
    if (sensorIdEl) {
        if (cfg.sensor_id !== undefined) {
            sensorIdEl.value = cfg.sensor_id;
        } else if (cfg.index !== undefined) {
            sensorIdEl.value = cfg.index;
        }
    }
    if (indexEl && cfg.index !== undefined) indexEl.value = cfg.index;
    if (widthEl) widthEl.value = cfg.width || defaults.width || 640;
    if (heightEl) heightEl.value = cfg.height || defaults.height || 480;
    if (fpsEl) fpsEl.value = cfg.fps || defaults.fps || 15;
    if (flipEl && cfg.flip_method !== undefined) flipEl.value = cfg.flip_method;
}

function readCameraSection(prefix, role, defaults) {
    function readNumber(el, fallback) {
        if (!el || el.value === "") return fallback;
        const num = parseInt(el.value, 10);
        return isFinite(num) ? num : fallback;
    }

    const enabledEl = document.getElementById("cfg-" + prefix + "-enabled");
    const typeEl = document.getElementById("cfg-" + prefix + "-type");
    const sensorIdEl = document.getElementById("cfg-" + prefix + "-sensor-id");
    const indexEl = document.getElementById("cfg-" + prefix + "-index");
    const widthEl = document.getElementById("cfg-" + prefix + "-w");
    const heightEl = document.getElementById("cfg-" + prefix + "-h");
    const fpsEl = document.getElementById("cfg-" + prefix + "-fps");
    const flipEl = document.getElementById("cfg-" + prefix + "-flip");

    const camera = {
        role: role,
        type: typeEl ? typeEl.value : (defaults.type || "csi"),
        enabled: enabledEl ? enabledEl.checked : true,
        width: readNumber(widthEl, defaults.width || 640),
        height: readNumber(heightEl, defaults.height || 480),
        fps: readNumber(fpsEl, defaults.fps || 15),
    };

    if (sensorIdEl && sensorIdEl.value !== "") {
        camera.sensor_id = readNumber(sensorIdEl, defaults.sensor_id);
    }
    if (indexEl && indexEl.value !== "") {
        camera.index = readNumber(indexEl, defaults.index);
    }
    if (flipEl && flipEl.value !== "") {
        camera.flip_method = readNumber(flipEl, defaults.flip_method);
    }

    if (camera.type === "csi" && camera.sensor_id === undefined && camera.index !== undefined) {
        camera.sensor_id = camera.index;
    }
    if (camera.type === "opencv" && camera.index === undefined && camera.sensor_id !== undefined) {
        camera.index = camera.sensor_id;
    }

    return camera;
}

function clearPreview() {
    const img = document.getElementById("live-preview");
    const placeholder = document.getElementById("preview-placeholder");
    if (img) {
        img.removeAttribute("src");
    }
    if (placeholder) {
        placeholder.style.display = "flex";
    }
    if (videoObjectUrl) {
        URL.revokeObjectURL(videoObjectUrl);
        videoObjectUrl = null;
    }
}

function closeVideoSocket() {
    videoSocketGeneration += 1;
    if (videoSocket) {
        try {
            videoSocket.onopen = null;
            videoSocket.onmessage = null;
            videoSocket.onerror = null;
            videoSocket.onclose = null;
            videoSocket.close();
        } catch (err) {
            console.warn("video close error", err);
        }
    }
    videoSocket = null;
    clearPreview();
}

function startLogPolling(carId) {
    if (logPollInterval) {
        clearInterval(logPollInterval);
        logPollInterval = null;
    }
    const list = document.getElementById("mission-log-list");
    if (list) {
        list.innerHTML = "<li class=\"log-empty\">Loading logs...</li>";
    }
    logCursorByCarId[carId] = logCursorByCarId[carId] || 0;
    pollCarLogs(carId);
    logPollInterval = setInterval(function () {
        pollCarLogs(carId);
    }, 1500);
}

async function pollCarLogs(carId) {
    if (!carId || selectedCarId !== carId) return;
    const cursor = logCursorByCarId[carId] || 0;
    try {
        const res = await fetch("/api/cars/" + encodeCarId(carId) + "/logs?since=" + cursor);
        if (!res.ok) return;
        const data = await res.json();
        const logs = data.logs || [];
        if (!logs.length) {
            if (cursor === 0) {
                const list = document.getElementById("mission-log-list");
                if (list && !list.children.length) {
                    list.innerHTML = "<li class=\"log-empty\">No logs yet.</li>";
                }
            }
            return;
        }
        appendLogs(logs);
        let maxTs = cursor;
        logs.forEach(function (entry) {
            if (entry.timestamp > maxTs) {
                maxTs = entry.timestamp;
            }
        });
        logCursorByCarId[carId] = maxTs;
    } catch (err) {
        console.warn("log poll failed", err);
    }
}

function appendLogs(logEntries) {
    const list = document.getElementById("mission-log-list");
    const panel = document.getElementById("mission-log-panel");
    if (!list) return;
    if (list.children.length === 1 && list.children[0].classList.contains("log-empty")) {
        list.innerHTML = "";
    }
    logEntries.forEach(function (entry) {
        const li = document.createElement("li");
        li.className = "log-entry";
        const ts = entry.timestamp ? new Date(entry.timestamp * 1000).toLocaleTimeString() : "--:--:--";
        li.textContent = "[" + ts + "] " + (entry.level || "INFO") + " " + (entry.message || "");
        list.appendChild(li);
    });
    while (list.children.length > 100) {
        list.removeChild(list.firstChild);
    }
    if (panel) {
        panel.scrollTop = panel.scrollHeight;
    }
}

function openTab(tabName) {
    document.querySelectorAll(".tab-content").forEach(function (el) {
        el.classList.remove("active");
    });
    document.querySelectorAll(".tab-btn").forEach(function (el) {
        el.classList.remove("active");
    });

    document.getElementById(tabName).classList.add("active");
    document.querySelector("button[onclick=\"openTab('" + tabName + "')\"]").classList.add("active");
}

function openClientTab(tabName) {
    document.querySelectorAll(".client-tab-pane").forEach(function (el) {
        el.style.display = "none";
        el.classList.remove("active");
    });
    document.querySelectorAll(".client-tab-btn").forEach(function (el) {
        el.classList.remove("active");
    });

    const content = document.getElementById("client-tab-" + tabName);
    if (content) {
        content.style.display = "block";
        content.classList.add("active");
    }

    const btns = document.querySelectorAll(".client-tab-btn");
    if (tabName === "overview" && btns[0]) btns[0].classList.add("active");
    if (tabName === "specs" && btns[1]) btns[1].classList.add("active");
}

async function fetchCars() {
    try {
        const res = await fetch("/api/cars");
        const data = await res.json();
        cars = data;
        renderFleetList();
        updateHomeStats();
        if (selectedCarId) {
            updateDetailView();
        }
        updateCharts();
    } catch (err) {
        console.error("Failed to fetch cars", err);
    }
}

function startPolling() {
    fetchCars();
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(fetchCars, pollIntervalMs);
}

async function fetchHostInfo() {
    try {
        const res = await fetch("/api/host-info");
        const data = await res.json();
        const el = document.getElementById("host-ip-info");
        if (el) el.innerText = "Server IP: " + data.ip + ":" + data.port;
    } catch (err) {
        console.error("Host info error:", err);
    }
}

function renderFleetList() {
    const list = document.getElementById("car-list");
    if (!list) return;
    list.innerHTML = "";

    cars.forEach(function (car) {
        const li = document.createElement("li");
        li.className = "car-item" + (selectedCarId === car.id ? " selected" : "");
        li.onclick = function () { selectCar(car.id); };

        let statusClass = "dot-gray";
        if (car.status === "online") statusClass = "dot-green";
        if (car.status === "stopped") statusClass = "dot-red";
        if (car.status === "offline") statusClass = "dot-gray";

        li.innerHTML = [
            "<div>",
            "<span class=\"status-dot " + statusClass + "\"></span>",
            "<strong>" + car.name + "</strong>",
            "</div>",
            "<small style=\"color:#888;\">" + car.ip + "</small>"
        ].join("");
        list.appendChild(li);
    });
}

async function selectCar(id) {
    closeVideoSocket();
    selectedCarId = id;
    renderFleetList();

    const emptyState = document.querySelector(".empty-state");
    const content = document.getElementById("car-detail-content");
    if (emptyState) emptyState.style.display = "none";
    if (content) content.style.display = "block";

    await fetch("/api/cars/" + encodeCarId(id) + "/status");
    await fetchCars();
    updateDetailView();
    openVideoSocket(id);
    startLogPolling(id);
}

function openVideoSocket(carId) {
    if (!carId) return;
    const generation = ++videoSocketGeneration;
    const socket = new WebSocket(getHostWebSocketBase("/ws/video/" + encodeCarId(carId)));
    socket.binaryType = "blob";
    videoSocket = socket;

    socket.onopen = function () {
        if (generation !== videoSocketGeneration || selectedCarId !== carId) return;
        setInlineStatus("Video connected.", "ok");
    };

    socket.onmessage = function (event) {
        if (generation !== videoSocketGeneration || selectedCarId !== carId) return;
        if (typeof event.data === "string") return;
        const img = document.getElementById("live-preview");
        const placeholder = document.getElementById("preview-placeholder");
        if (!img) return;
        if (videoObjectUrl) {
            URL.revokeObjectURL(videoObjectUrl);
        }
        videoObjectUrl = URL.createObjectURL(event.data);
        img.src = videoObjectUrl;
        if (placeholder) {
            placeholder.style.display = "none";
        }
    };

    socket.onerror = function () {
        if (generation !== videoSocketGeneration || selectedCarId !== carId) return;
        setInlineStatus("Video unavailable.", "warn");
    };

    socket.onclose = function () {
        if (generation !== videoSocketGeneration || selectedCarId !== carId) return;
        setInlineStatus("Video disconnected.", "warn");
        clearPreview();
    };
}

function updateMissionView(mission) {
    const state = mission || {};
    const missionState = state.state || "IDLE";
    const stopReason = state.stop_reason || "";
    const detectorStatus = state.tag_detector_status || "Unknown";

    setBadge("mission-state-badge", missionState, missionState === "FAULT_STOP" ? "danger" : missionState === "COMPLETE" ? "success" : missionState === "RUNNING" || missionState === "APPROACH_GOAL" ? "ok" : "neutral");
    setBadge("stop-reason-badge", stopReason ? stopReason : "No stop reason", stopReason ? "warning" : "neutral");
    setText("mission-route", state.route_name || "expo_route");
    setText("mission-tag-status", detectorStatus);
    setText("obstacle-distance", formatDistanceMeters(state.obstacle_distance_m));
    setText("last-tag", state.last_tag_id === null || state.last_tag_id === undefined ? "N/A" : String(state.last_tag_id));
    setText("route-start-seen", formatBool(state.start_tag_seen));
    setText("route-checkpoint-seen", formatBool(state.checkpoint_seen));
    setText("route-goal-seen", formatBool(state.goal_seen));
    setText("route-expected-next", state.expected_next_tag === null || state.expected_next_tag === undefined ? "N/A" : String(state.expected_next_tag));
}

function updateDetailView() {
    const car = getSelectedCar();
    if (!car) return;

    document.getElementById("detail-name").innerText = car.name || "Car";
    document.getElementById("detail-status").innerText = (car.status || "UNKNOWN").toUpperCase();
    document.getElementById("detail-ip").innerText = (car.ip || "") + ":" + (car.port || "");

    const geo = car.geo || {};
    document.getElementById("detail-geo").innerText = geo.city ? (geo.city + ", " + geo.country) : "Unknown / Local";

    const d = car.details || {};
    if (d.error) {
        document.getElementById("detail-activity").innerHTML = "<span style=\"color:#ff8a8a\">Error: " + d.error + "</span>";
    } else {
        const running = d.running ? "Running" : "Stopped";
        document.getElementById("detail-activity").innerText = running + (d.paused ? " (Paused)" : "");
    }

    const fps = d.state && d.state.fps ? d.state.fps : 0;
    document.getElementById("detail-fps").innerText = String(fps);

    const state = d.state || {};
    const sensorRig = state.sensors || state.sensor_rig || {};
    renderSensorRig(sensorRig);

    const mission = state.mission ? state.mission : {};
    updateMissionView(mission);

    const lastAction = state.last_action ? state.last_action : null;
    if (lastAction) {
        const steer = lastAction.steer !== undefined ? Number(lastAction.steer).toFixed(2) : "n/a";
        const throttle = lastAction.throttle !== undefined ? Number(lastAction.throttle).toFixed(2) : "n/a";
        setText("last-action", "steer " + steer + ", throttle " + throttle);
    } else {
        setText("last-action", "N/A");
    }

    const loc = state.location || {};
    const imu = loc.imu || (sensorRig.sidecar_depth_imu && sensorRig.sidecar_depth_imu.imu_data) || {};
    document.getElementById("imu-accel").innerText = imu.accel ? formatVector(imu.accel) : "N/A";
    document.getElementById("imu-gyro").innerText = imu.gyro ? formatVector(imu.gyro) : "N/A";

    const detList = document.getElementById("detection-list");
    if (detList) {
        detList.innerHTML = "";
        const dets = state.detections ? state.detections : [];
        if (!dets.length) {
            detList.innerHTML = "<li>No objects detected</li>";
        } else {
            dets.forEach(function (obj) {
                const li = document.createElement("li");
                const dist = obj.distance !== undefined && obj.distance !== null ? formatDistanceMeters(Number(obj.distance) > 20 ? Number(obj.distance) / 1000.0 : Number(obj.distance)) : "N/A";
                const conf = obj.conf !== undefined && obj.conf !== null ? (Number(obj.conf) * 100).toFixed(0) + "%" : "N/A";
                li.textContent = "Class: " + obj.class + " | Dist: " + dist + " | Conf: " + conf;
                detList.appendChild(li);
            });
        }
    }

    const specs = state.specs ? state.specs : {};
    document.getElementById("spec-device").innerText = specs.device || "Unknown";
    document.getElementById("spec-cpu").innerText = specs.cpu_ram || "Unknown";

    let camText = "None";
    if (specs.cameras && Array.isArray(specs.cameras)) {
        camText = specs.cameras.map(function (c) {
            return cameraSummary(c);
        }).join(", ");
    } else if (specs.cameras) {
        camText = specs.cameras;
    }
    document.getElementById("spec-cameras").innerText = camText;
    document.getElementById("spec-inference").innerText = specs.inference || "Unknown";
    document.getElementById("spec-resnet").innerText = specs.resnet_version || "Unknown";
    document.getElementById("spec-yolo").innerText = specs.yolo_version || "Unknown";
}

function updateHomeStats() {
    const total = cars.length;
    const active = cars.filter(function (c) { return c.status === "online"; }).length;
    document.getElementById("stat-total").innerText = String(total);
    document.getElementById("stat-active").innerText = String(active);
}

async function addTestClient() {
    try {
        const res = await fetch("/api/test-client", { method: "POST" });
        if (!res.ok) throw new Error(await res.text());
        setInlineStatus("Test client added.", "ok");
        fetchCars();
    } catch (err) {
        setInlineStatus("Add test client failed.", "error");
        console.error(err);
    }
}

async function addCar() {
    const name = document.getElementById("new-name").value;
    const ip = document.getElementById("new-ip").value;
    const port = document.getElementById("new-port").value;

    if (!name || !ip) {
        setInlineStatus("Fill all fields.", "warn");
        return;
    }

    try {
        const res = await fetch("/api/cars", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: name, ip: ip, port: parseInt(port, 10) })
        });
        if (!res.ok) throw new Error(await res.text());
        closeAddModal();
        setInlineStatus("Car added.", "ok");
        fetchCars();
    } catch (err) {
        setInlineStatus("Add car failed.", "error");
        console.error(err);
    }
}

async function deleteCar() {
    if (!selectedCarId) return;
    if (!confirm("Remove this car?")) return;
    try {
        await fetch("/api/cars/" + encodeCarId(selectedCarId), { method: "DELETE" });
        closeVideoSocket();
        if (logPollInterval) {
            clearInterval(logPollInterval);
            logPollInterval = null;
        }
        selectedCarId = null;
        document.getElementById("car-detail-content").style.display = "none";
        document.querySelector(".empty-state").style.display = "block";
        setInlineStatus("Car removed.", "ok");
        fetchCars();
    } catch (err) {
        setInlineStatus("Remove car failed.", "error");
        console.error(err);
    }
}

async function controlAction(action) {
    if (!selectedCarId) return;
    try {
        const res = await fetch("/api/cars/" + encodeCarId(selectedCarId) + "/" + action, { method: "POST" });
        if (!res.ok) throw new Error(await res.text());
        setInlineStatus(action.toUpperCase() + " command sent.", "ok");
    } catch (err) {
        setInlineStatus(action.toUpperCase() + " failed.", "error");
        console.error(err);
    }
}

async function updateSettings() {
    if (!selectedCarId) return;
    const modeEl = document.getElementById("tune-mode");
    const throttleEl = document.getElementById("tune-throttle");
    if (!modeEl || !throttleEl) return;

    try {
        const res = await fetch("/api/cars/" + encodeCarId(selectedCarId) + "/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                throttle_mode: modeEl.value,
                fixed_throttle_value: parseFloat(throttleEl.value)
            })
        });
        if (!res.ok) throw new Error(await res.text());
        setInlineStatus("Tuning updated.", "ok");
    } catch (err) {
        setInlineStatus("Tuning update failed.", "error");
        console.error(err);
    }
}

function showAddModal() { document.getElementById("add-modal").style.display = "flex"; }
function closeAddModal() { document.getElementById("add-modal").style.display = "none"; }

function openConfigModal() {
    if (!selectedCarId) return;
    const car = getSelectedCar();
    if (!car) return;

    document.getElementById("config-modal").style.display = "flex";

    const cfg = (car.details && car.details.config) ? car.details.config : {};
    const cameras = Array.isArray(cfg.cameras) ? cfg.cameras : [];
    const primary = getCameraByRole(cameras, "primary_rgb") || cameras[0] || { type: "csi", width: 640, height: 480, fps: 15, enabled: true };
    const sidecar = getCameraByRole(cameras, "sidecar_depth_imu") || (primary.type === "realsense" ? primary : { type: "realsense", width: 640, height: 480, fps: 15, enabled: false });
    const rear = getCameraByRole(cameras, "rear_preview") || { type: "csi", sensor_id: 1, width: 640, height: 480, fps: 15, flip_method: 2, enabled: false };

    populateCameraSection("primary", primary, { type: "csi", width: 640, height: 480, fps: 15, enabled: true, flip_method: 2 });
    populateCameraSection("sidecar", sidecar, { type: "realsense", width: 640, height: 480, fps: 15, enabled: primary.type === "realsense" });
    populateCameraSection("rear", rear, { type: "csi", sensor_id: 1, width: 640, height: 480, fps: 15, enabled: false, flip_method: 2 });
    document.getElementById("cfg-ctl-type").value = cfg.control_model_type || "tensorrt";
    document.getElementById("cfg-ctl-arch").value = cfg.architecture || "resnet101";
    document.getElementById("cfg-ctl-path").value = cfg.control_model || "";
    document.getElementById("cfg-det-path").value = cfg.detection_model || "";
    document.getElementById("cfg-device").value = cfg.device || "cuda";
    document.getElementById("cfg-actions").value = (cfg.action_loop || ["control", "api"]).join(",");
}

function closeConfigModal() {
    document.getElementById("config-modal").style.display = "none";
}

function resetConfigToDefaults() {
    populateCameraSection("primary", { type: "csi", width: 640, height: 480, fps: 15, enabled: true, sensor_id: 0, flip_method: 2 }, { type: "csi", width: 640, height: 480, fps: 15, enabled: true, flip_method: 2 });
    populateCameraSection("sidecar", { type: "realsense", width: 640, height: 480, fps: 15, enabled: true }, { type: "realsense", width: 640, height: 480, fps: 15, enabled: true });
    populateCameraSection("rear", { type: "csi", sensor_id: 1, width: 640, height: 480, fps: 15, flip_method: 2, enabled: false }, { type: "csi", sensor_id: 1, width: 640, height: 480, fps: 15, flip_method: 2, enabled: false });
    document.getElementById("cfg-ctl-type").value = "tensorrt";
    document.getElementById("cfg-ctl-arch").value = "resnet101";
    document.getElementById("cfg-ctl-path").value = "/home/jetson/jetracer_run/checkpoints/checkpoints/model_7_resnet101/best_model_trt.pth";
    document.getElementById("cfg-det-path").value = "";
    document.getElementById("cfg-device").value = "cuda";
    document.getElementById("cfg-actions").value = "control,detection,api";
}

async function deployConfig() {
    if (!selectedCarId) return;

    const current = getSelectedCar();
    const currentMission = current && current.details && current.details.config ? current.details.config.mission : null;
    const primary = readCameraSection("primary", "primary_rgb", { type: "csi", width: 640, height: 480, fps: 15 });
    const sidecar = readCameraSection("sidecar", "sidecar_depth_imu", { type: "realsense", width: 640, height: 480, fps: 15 });
    const rear = readCameraSection("rear", "rear_preview", { type: "csi", sensor_id: 1, width: 640, height: 480, fps: 15, flip_method: 2 });
    const preprocessProfile = primary.enabled && primary.type === "csi" ? "cam0_fisheye_v1" : "legacy_resize_v0";
    const config = {
        device: document.getElementById("cfg-device").value,
        architecture: document.getElementById("cfg-ctl-arch").value,
        preprocess_profile: preprocessProfile,
        cameras: [primary, sidecar, rear],
        control_model_type: document.getElementById("cfg-ctl-type").value,
        control_model: document.getElementById("cfg-ctl-path").value,
        detection_model: document.getElementById("cfg-det-path").value,
        action_loop: document.getElementById("cfg-actions").value.split(",").map(function (s) { return s.trim(); }).filter(Boolean),
        ip: "0.0.0.0",
        port: 8000
    };
    if (currentMission) {
        config.mission = currentMission;
    }

    try {
        const res = await fetch("/api/cars/" + encodeCarId(selectedCarId) + "/config", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ config: config })
        });
        const data = await res.json();
        if (data.status === "configured" || data.status === "sent_via_ws" || data.status === "sent_via_http") {
            setInlineStatus("Configuration deployed.", "ok");
            closeConfigModal();
            fetchCars();
        } else {
            setInlineStatus("Deploy failed.", "error");
            console.error(data);
        }
    } catch (err) {
        setInlineStatus("Deploy failed.", "error");
        console.error(err);
    }
}

function saveGlobalSettings() {
    const poll = document.getElementById("setting-poll").value;
    const pollMs = parseInt(poll, 10);
    if (isFinite(pollMs) && pollMs > 0) {
        pollIntervalMs = pollMs;
    }
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(fetchCars, pollIntervalMs);
    setInlineStatus("Settings saved.", "ok");
}

function updateCharts() {
    const canvas = document.getElementById("statusChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const online = cars.filter(function (c) { return c.status === "online"; }).length;
    const stopped = cars.filter(function (c) { return c.status === "stopped"; }).length;
    const offline = cars.filter(function (c) { return c.status === "offline" || c.status === "unknown"; }).length;

    if (statusChart) {
        statusChart.data.datasets[0].data = [online, stopped, offline];
        statusChart.update();
    } else {
        statusChart = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: ["Running", "Stopped", "Offline"],
                datasets: [{
                    data: [online, stopped, offline],
                    backgroundColor: ["#28a745", "#dc3545", "#6c757d"]
                }]
            },
            options: { responsive: true, plugins: { legend: { position: "bottom" } } }
        });
    }
}

window.onclick = function (event) {
    if (event.target === document.getElementById("add-modal")) {
        closeAddModal();
    }
    if (event.target === document.getElementById("config-modal")) {
        closeConfigModal();
    }
};

startPolling();
fetchHostInfo();
