// State
let cars = [];
let selectedCarId = null;
let pollInterval = null;

// --- Tab Logic ---
function openTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    
    document.getElementById(tabName).classList.add('active');
    document.querySelector(`button[onclick="openTab('${tabName}')"]`).classList.add('active');
}

function openClientTab(tabName) {
    document.querySelectorAll('.client-tab-pane').forEach(el => {
        el.style.display = 'none';
        el.classList.remove('active');
    });
    document.querySelectorAll('.client-tab-btn').forEach(el => el.classList.remove('active'));
    
    const content = document.getElementById(`client-tab-${tabName}`);
    if(content) {
        content.style.display = 'block';
        content.classList.add('active');
    }
    
    // Find button to highlight
    const btns = document.querySelectorAll('.client-tab-btn');
    if (tabName === 'overview') btns[0].classList.add('active');
    if (tabName === 'specs') btns[1].classList.add('active');
}

// --- Poll Data ---
async function fetchCars() {
    try {
        const res = await fetch('/api/cars');
        const data = await res.json();
        cars = data;
        renderFleetList();
        updateHomeStats();
        if (selectedCarId) {
            updateDetailView();
        }
        updateCharts();
    } catch (e) {
        console.error("Failed to fetch cars", e);
    }
}

function startPolling() {
    fetchCars();
    pollInterval = setInterval(fetchCars, 2000); // 2s poll
}

async function fetchHostInfo() {
    try {
        const res = await fetch('/api/host-info');
        const data = await res.json();
        const el = document.getElementById('host-ip-info');
        if (el) el.innerText = "Server IP: " + data.ip + ":" + data.port;
    } catch (e) {
        console.error("Host info error:", e);
    }
}

// --- Render Logic ---
function renderFleetList() {
    const list = document.getElementById('car-list');
    list.innerHTML = '';
    
    cars.forEach(car => {
        const li = document.createElement('li');
        li.className = `car-item ${selectedCarId === car.id ? 'selected' : ''}`;
        li.onclick = () => selectCar(car.id);
        
        let statusClass = 'dot-gray'; // unknown
        if (car.status === 'online') statusClass = 'dot-green';
        if (car.status === 'stopped') statusClass = 'dot-red';
        if (car.status === 'offline') statusClass = 'dot-gray';
        
        li.innerHTML = `
            <div>
                <span class="status-dot ${statusClass}"></span>
                <strong>${car.name}</strong>
            </div>
            <small style="color:#888;">${car.ip}</small>
        `;
        list.appendChild(li);
    });
}

function selectCar(id) {
    selectedCarId = id;
    renderFleetList(); // update active state
    
    document.querySelector('.empty-state').style.display = 'none';
    document.getElementById('car-detail-content').style.display = 'block';
    
    // Trigger immediate refresh of this specific car's status to get live data
    fetch(`/api/cars/${id}/status`); 
    
    updateDetailView();
}

function updateDetailView() {
    if (!selectedCarId) return;
    const car = cars.find(c => c.id === selectedCarId);
    if (!car) return;

    document.getElementById('detail-name').innerText = car.name;
    document.getElementById('detail-status').innerText = car.status.toUpperCase();
    document.getElementById('detail-ip').innerText = `${car.ip}:${car.port}`;
    
    // Geo
    const geo = car.geo || {};
    document.getElementById('detail-geo').innerText = 
        geo.city ? `${geo.city}, ${geo.country}` : 'Unknown / Local';

    // Details from deep state
    const d = car.details || {};
    
    if (d.error) {
        document.getElementById('detail-activity').innerHTML = `<span style="color:red">Error: ${d.error}</span>`;
    } else {
        const running = d.running ? "Running" : "Stopped";
        document.getElementById('detail-activity').innerText = running + (d.paused ? " (Paused)" : "");
    }
    
    const fps = d.state?.fps || 0;
    document.getElementById('detail-fps').innerText = fps;

    // IMU Data
    const loc = d.state?.location || {};
    const imu = loc.imu || {};
    if (imu.accel) {
        document.getElementById('imu-accel').innerText = 
            `X:${imu.accel[0].toFixed(2)} Y:${imu.accel[1].toFixed(2)} Z:${imu.accel[2].toFixed(2)}`;
    } else {
        document.getElementById('imu-accel').innerText = "N/A";
    }
    if (imu.gyro) {
        document.getElementById('imu-gyro').innerText = 
            `X:${imu.gyro[0].toFixed(2)} Y:${imu.gyro[1].toFixed(2)} Z:${imu.gyro[2].toFixed(2)}`;
    } else {
        document.getElementById('imu-gyro').innerText = "N/A";
    }

    // Current settings
    // If we just loaded, we might want to sync UI with current throttle...
    // But since backend doesn't always send settings back in basic status, we rely on user input.

    // Detections
    const detList = document.getElementById('detection-list');
    detList.innerHTML = '';
    const dets = d.state?.detections || [];
    if (dets.length === 0) {
        detList.innerHTML = '<li>No objects detected</li>';
    } else {
        dets.forEach(obj => {
            const dist = obj.distance ? (obj.distance/1000).toFixed(2) + 'm' : 'N/A';
            const li = document.createElement('li');
            li.innerHTML = `Class: <strong>${obj.class}</strong> | Dist: <strong>${dist}</strong> | Conf: ${(obj.conf*100).toFixed(0)}%`;
            detList.appendChild(li);
        });
    }

    // Call SLAM Draw
    if (d.state?.location) {
       // Draw Map update
       // We need to pass the whole state or just location
       // Let's pass the car object or details
       drawSlamMap(car);
    }

    // Update Specs Tab
    const specs = d.state?.specs || {}; 
    document.getElementById('spec-device').innerText = specs.device || 'Unknown';
    document.getElementById('spec-cpu').innerText = specs.cpu_ram || 'Unknown';
    
    // Format cameras nicely
    let camText = 'None';
    if (specs.cameras && Array.isArray(specs.cameras)) {
        camText = specs.cameras.map(c => `${c.type} (${c.width}x${c.height})`).join(', ');
    } else if (specs.cameras) {
        camText = specs.cameras;
    }
    document.getElementById('spec-cameras').innerText = camText;
    
    document.getElementById('spec-inference').innerText = specs.inference || 'Unknown';
    document.getElementById('spec-resnet').innerText = specs.resnet_version || 'Unknown';
    document.getElementById('spec-yolo').innerText = specs.yolo_version || 'Unknown';
}

function updateHomeStats() {
    const total = cars.length;
    const active = cars.filter(c => c.status === 'online').length;
    document.getElementById('stat-total').innerText = total;
    document.getElementById('stat-active').innerText = active;
}

// --- Actions ---
async function addTestClient() {
    try {
        const res = await fetch('/api/test-client', { method: 'POST' });
        if(!res.ok) throw new Error(await res.text());
        fetchCars();
    } catch(e) {
        alert("Error adding test client: " + e.message);
    }
}

// Deprecated: was for manual add
async function addCar() {
    const name = document.getElementById('new-name').value;
    const ip = document.getElementById('new-ip').value;
    const port = document.getElementById('new-port').value;
    
    if (!name || !ip) return alert("Fill all fields");
    
    await fetch('/api/cars', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name, ip, port: parseInt(port)})
    });
    
    closeAddModal();
    fetchCars();
}

async function deleteCar() {
    if(!selectedCarId) return;
    if(!confirm("Remove this car?")) return;
    await fetch(`/api/cars/${selectedCarId}`, {method: 'DELETE'});
    selectedCarId = null;
    document.getElementById('car-detail-content').style.display = 'none';
    document.querySelector('.empty-state').style.display = 'block';
    fetchCars();
}

async function controlAction(action) {
    if(!selectedCarId) return;
    await fetch(`/api/cars/${selectedCarId}/${action}`, {method: 'POST'});
    // Provide immediate feedback
    alert(`Sent ${action.toUpperCase()} command.`);
}

async function updateSettings() {
    if(!selectedCarId) return;
    const mode = document.getElementById('tune-mode').value;
    const throttle = parseFloat(document.getElementById('tune-throttle').value);
    
    await fetch(`/api/cars/${selectedCarId}/settings`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            throttle_mode: mode,
            fixed_throttle_value: throttle
        })
    });
}

// --- Modal ---
function showAddModal() { document.getElementById('add-modal').style.display = 'flex'; }
function closeAddModal() { document.getElementById('add-modal').style.display = 'none'; }

// --- Config Modal ---
function openConfigModal() {
    if(!selectedCarId) return;
    const car = cars.find(c => c.id === selectedCarId);
    if (!car) return;

    document.getElementById('config-modal').style.display = 'flex';
    
    // Populate with current car configuration if available
    const cfg = car.details?.config || {};
    if (Object.keys(cfg).length > 0) {
        if (cfg.cameras && cfg.cameras.length > 0) {
            const cam = cfg.cameras[0];
            document.getElementById('cfg-cam-type').value = cam.type || 'realsense';
            document.getElementById('cfg-cam-w').value = cam.width || 640;
            document.getElementById('cfg-cam-h').value = cam.height || 480;
            document.getElementById('cfg-cam-fps').value = cam.fps || 15;
        }
        document.getElementById('cfg-ctl-type').value = cfg.control_model_type || 'tensorrt';
        document.getElementById('cfg-ctl-arch').value = cfg.architecture || 'resnet101';
        document.getElementById('cfg-ctl-path').value = cfg.control_model || '';
        document.getElementById('cfg-det-path').value = cfg.detection_model || '';
        document.getElementById('cfg-device').value = cfg.device || 'cuda';
        document.getElementById('cfg-actions').value = (cfg.action_loop || ['control', 'api']).join(',');
    }
}

function closeConfigModal() {
    document.getElementById('config-modal').style.display = 'none';
}

function resetConfigToDefaults() {
    document.getElementById('cfg-cam-type').value = 'realsense';
    document.getElementById('cfg-cam-w').value = '640';
    document.getElementById('cfg-cam-h').value = '480';
    document.getElementById('cfg-cam-fps').value = '15';
    document.getElementById('cfg-ctl-type').value = 'tensorrt';
    document.getElementById('cfg-ctl-arch').value = 'resnet101';
    document.getElementById('cfg-ctl-path').value = '/home/jetson/jetracer_run/checkpoints/checkpoints/model_7_resnet101/best_model_trt.pth';
    document.getElementById('cfg-det-path').value = '';
    document.getElementById('cfg-device').value = 'cuda';
    document.getElementById('cfg-actions').value = 'control,detection,api';
}

async function deployConfig() {
    if(!selectedCarId) return;
    
    const config = {
        device: document.getElementById('cfg-device').value,
        architecture: document.getElementById('cfg-ctl-arch').value,
        cameras: [{
            type: document.getElementById('cfg-cam-type').value,
            width: parseInt(document.getElementById('cfg-cam-w').value),
            height: parseInt(document.getElementById('cfg-cam-h').value),
            fps: parseInt(document.getElementById('cfg-cam-fps').value),
            index: 0
        }],
        control_model_type: document.getElementById('cfg-ctl-type').value,
        control_model: document.getElementById('cfg-ctl-path').value,
        detection_model: document.getElementById('cfg-det-path').value,
        action_loop: document.getElementById('cfg-actions').value.split(',').map(s=>s.trim()),
        ip: "0.0.0.0", // overridden by server proxy
        port: 8000,
        password: "changeme"
    };

    try {
        const res = await fetch(`/api/cars/${selectedCarId}/config`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({config: config})
        });
        const data = await res.json();
        // Treat both "configured" (HTTP) and "sent_via_ws" (WebSocket) as success
        if(data.status === 'configured' || data.status === 'sent_via_ws') {
            alert("Configuration Deployed Successfully!");
            closeConfigModal();
        } else {
            alert("Error: " + JSON.stringify(data));
        }
    } catch(e) {
        alert("Deploy failed: " + e);
    }
}

// --- Global Settings ---
function saveGlobalSettings() {
    const poll = document.getElementById('setting-poll').value;
    // Update poll interval
    if (pollInterval) clearInterval(pollInterval);
    setInterval(fetchCars, parseInt(poll));
    alert("Settings Saved");
}

// --- Charts ---
let statusChart = null;

function updateCharts() {
    // Simple Status Pie Chart
    const ctx = document.getElementById('statusChart').getContext('2d');
    const online = cars.filter(c => c.status === 'online').length;
    const stopped = cars.filter(c => c.status === 'stopped').length;
    const offline = cars.filter(c => c.status === 'offline' || c.status === 'unknown').length;
    
    if (statusChart) {
        statusChart.data.datasets[0].data = [online, stopped, offline];
        statusChart.update();
    } else {
        statusChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Running', 'Stopped', 'Offline'],
                datasets: [{
                    data: [online, stopped, offline],
                    backgroundColor: ['#28a745', '#dc3545', '#6c757d']
                }]
            },
            options: { responsive: true, plugins: { legend: { position: 'bottom' } } }
        });
    }
}

// --- SLAM Frontend ---
let mapScale = 50.0; // pixels per meter
let mapCanvas = null;
let mapCtx = null;
let selectedDestination = null;

function setupMapCanvas() {
    mapCanvas = document.getElementById('slam-map');
    if (!mapCanvas) return;
    
    mapCtx = mapCanvas.getContext('2d');
    
    // Support clicking to set destination
    mapCanvas.addEventListener('mousedown', async (e) => {
        if (!selectedCarId) return;
        
        const rect = mapCanvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;
        
        // Convert screen to world
        // Center is (width/2, height/2)
        // World 0,0 is at center
        // X is Right, Y is Up (in screen, Y is Down)
        const centerX = mapCanvas.width / 2;
        const centerY = mapCanvas.height / 2;
        
        const worldX = (clickX - centerX) / mapScale;
        const worldY = -(clickY - centerY) / mapScale; // invert Y
        
        selectedDestination = {x: worldX, y: worldY};
        
        // Send command
        try {
            await fetch(`/api/cars/${selectedCarId}/navigate`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({x: worldX, y: worldY})
            });
            console.log("Navigating to", worldX, worldY);
        } catch(e) {
            console.error(e);
        }
    });
}

function drawSlamMap(car) {
    if (!mapCtx) setupMapCanvas();
    if (!mapCtx) return;
    
    const w = mapCanvas.width;
    const h = mapCanvas.height;
    const cx = w / 2;
    const cy = h / 2;
    
    // Clear
    mapCtx.fillStyle = "#000";
    mapCtx.fillRect(0, 0, w, h);
    
    // Draw Grid (1m lines)
    mapCtx.strokeStyle = "#333";
    mapCtx.lineWidth = 1;
    mapCtx.beginPath();
    
    // Vertical lines
    for(let x=-10; x<=10; x++) {
        const sx = cx + x * mapScale;
        mapCtx.moveTo(sx, 0); mapCtx.lineTo(sx, h);
    }
    // Horizontal lines
    for(let y=-10; y<=10; y++) {
        const sy = cy - y * mapScale;
        mapCtx.moveTo(0, sy); mapCtx.lineTo(w, sy);
    }
    mapCtx.stroke();
    
    // Origin
    mapCtx.fillStyle = "#555";
    mapCtx.beginPath();
    mapCtx.arc(cx, cy, 3, 0, 2*Math.PI);
    mapCtx.fill();
    
    // Get state
    const state = car.details?.state || {};
    const loc = state.location || {x: 0, y: 0, theta: 0};
    
    document.getElementById('slam-coords').innerText = 
        `X: ${loc.x.toFixed(2)} Y: ${loc.y.toFixed(2)} θ: ${(loc.theta * 180 / Math.PI).toFixed(0)}°`;
    
    // Draw Trajectory
    if (loc.trajectory && loc.trajectory.length > 0) {
        mapCtx.strokeStyle = "#0ff";
        mapCtx.lineWidth = 2;
        mapCtx.beginPath();
        loc.trajectory.forEach((pt, i) => {
            const sx = cx + pt[0] * mapScale;
            const sy = cy - pt[1] * mapScale;
            if (i===0) mapCtx.moveTo(sx, sy);
            else mapCtx.lineTo(sx, sy);
        });
        // Draw line to current
        const currSx = cx + loc.x * mapScale;
        const currSy = cy - loc.y * mapScale;
        mapCtx.lineTo(currSx, currSy);
        mapCtx.stroke();
    }
    
    // Draw Destination
    if (selectedDestination) {
        const dx = cx + selectedDestination.x * mapScale;
        const dy = cy - selectedDestination.y * mapScale;
        
        mapCtx.strokeStyle = "#f0f"; // Magenta target
        mapCtx.beginPath();
        mapCtx.arc(dx, dy, 5, 0, 2*Math.PI);
        mapCtx.stroke();
        
        // Line to target
        mapCtx.setLineDash([5, 5]);
        mapCtx.beginPath();
        mapCtx.moveTo(cx + loc.x * mapScale, cy - loc.y * mapScale);
        mapCtx.lineTo(dx, dy);
        mapCtx.stroke();
        mapCtx.setLineDash([]);
    }
    
    // Draw Car
    const carSx = cx + loc.x * mapScale;
    const carSy = cy - loc.y * mapScale;
    
    mapCtx.save();
    mapCtx.translate(carSx, carSy);
    mapCtx.rotate(-loc.theta); // -theta because canvas Y is down? No, standard rotate is CW. Theta is CCW normally.
    // If loc.theta is standard math angle (CCW from X), and Canvas Y is inverted...
    // World: +Y is Up. Screen: +Y is Down.
    // Math: +Angle is CCW. Screen: +Angle is CW.
    // So to draw CCW angle on inverted Y, we need -angle.
    
    // Car Body
    mapCtx.fillStyle = "#0f0";
    mapCtx.fillRect(-10, -5, 20, 10); // 20px long car
    
    // Direction Indicator
    mapCtx.fillStyle = "#fff";
    mapCtx.fillRect(5, -5, 5, 10); // Front lights
    
    mapCtx.restore();
}

async function cancelNavigation() {
    if (!selectedCarId) return;
    selectedDestination = null;
    await fetch(`/api/cars/${selectedCarId}/navigate/cancel`, {method: 'POST'});
}

// Init
window.onclick = function(event) {
    if (event.target == document.getElementById('add-modal')) {
        closeAddModal();
    }
    if (event.target == document.getElementById('config-modal')) {
        closeConfigModal();
    }
}

startPolling();
fetchHostInfo();
