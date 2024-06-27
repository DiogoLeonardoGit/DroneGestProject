const WebSocket = require('ws');

const wss = new WebSocket.Server({ port: 8081 });

wss.on('connection', ws => {
    ws.on('message', message => {
        console.log('received:', message);
        // Broadcast the message to all clients
        wss.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(message);
            }
        });
    });

    ws.send('WebSocket server connected');
});

console.log('WebSocket server running on ws://localhost:8081');
