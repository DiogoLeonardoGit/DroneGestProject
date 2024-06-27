// Set up the scene, camera, and renderer
const scene = new THREE.Scene();

// Load a background image
const textureLoader = new THREE.TextureLoader();
textureLoader.load('textures/background2.jpg', function (texture) {
    const bgGeometry = new THREE.PlaneGeometry(60, 30); // Adjust the size as needed
    const bgMaterial = new THREE.MeshBasicMaterial({ map: texture });
    const bgMesh = new THREE.Mesh(bgGeometry, bgMaterial);

    // Position the background plane
    bgMesh.position.z = -10; // Move it back behind all other objects
    scene.add(bgMesh);
}, undefined, function (error) {
    console.error('Error loading background image:', error);
});

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Adjust the renderer and camera on window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// Add lighting
const light = new THREE.AmbientLight(0x808080); // Stronger ambient light
scene.add(light);
const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
scene.add(directionalLight);

// Load the GLTF model and add a ground plane for reference
const loader = new THREE.GLTFLoader();
let drone;
loader.load('drone-model.glb', function (gltf) {
    drone = gltf.scene;
    scene.add(drone);
    drone.position.set(0, 0, 0);  // Position the drone at the origin
    drone.scale.set(1, 1, 1);  // Scale the drone if necessary

    // Load and apply texture to the drone
    const droneTexture = new THREE.TextureLoader().load('textures/droneTex.png'); // Replace with your texture path
    drone.traverse((node) => {
        if (node.isMesh) {
            node.castShadow = true;
            node.receiveShadow = true;
            node.material = new THREE.MeshStandardMaterial({ map: droneTexture }); // Apply the texture
        }
    });

    // Add a ground plane for reference
    const geometry = new THREE.PlaneGeometry(100, 100);
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00, side: THREE.DoubleSide });
    const plane = new THREE.Mesh(geometry, material);
    plane.rotation.x = Math.PI / 2;  // Rotate to make it horizontal
    scene.add(plane);

    // Set camera position
    camera.position.z = 7.5;

    // WebSocket setup
    const socket = new WebSocket('ws://localhost:8081'); // Replace with your WebSocket server address

    socket.onopen = function () {
        console.log('WebSocket connection opened');
    };

    socket.onmessage = function (event) {
        const command = event.data.toLowerCase();
        if (drone) {
            switch (command) {
                case 'up':
                    drone.position.y += 1;
                    break;
                case 'down':
                    drone.position.y -= 1;
                    break;
                case 'left':
                    drone.position.x -= 1;
                    break;
                case 'right':
                    drone.position.x += 1;
                    break;
                case 'forward':
                    drone.position.z -= 1;
                    break;
                case 'backward':
                    drone.position.z += 1;
                    break;
                default:
                    console.log('Unknown command:', command);
            }
        }
    };

    socket.onclose = function () {
        console.log('WebSocket connection closed');
    };

    socket.onerror = function (error) {
        console.error('WebSocket error:', error);
    };

    // Render loop
    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    }

    animate(); // Start the render loop after setting up everything
}, undefined, function (error) {
    console.error('Error loading drone model:', error);
});

// Add zoom in/out functionality
window.addEventListener('wheel', (event) => {
    const zoomSpeed = 1.1;
    if (event.deltaY < 0) {
        camera.position.z /= zoomSpeed;
    } else {
        camera.position.z *= zoomSpeed;
    }
});
