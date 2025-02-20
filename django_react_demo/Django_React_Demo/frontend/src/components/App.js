import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './HomePage';
import RoomJoinPage from './RoomJoinPage';
import CreateRoomPage from './CreateRoomPage';
import Room from './Room';

function App() {
    return (
        <Router>
            <div className='center'>
                <Routes>
                    <Route path="/" element={<HomePage />} />
                    <Route path="/join" element={<RoomJoinPage />} />
                    <Route path="/create" element={<CreateRoomPage />} />
                    <Route path='/room/:roomCode' element={<Room />} />
                </Routes>
            </div>
        </Router>
    );
}

const appDiv = document.getElementById("app");
const root = createRoot(appDiv); // This is a new method in React 18 that allows you to render the app in a concurrent mode

// Render the app component
root.render(<App />);

export default App;
