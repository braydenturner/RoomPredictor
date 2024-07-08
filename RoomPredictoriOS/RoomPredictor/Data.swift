//
//  Data.swift
//  RoomPredictor
//
//  Created by Brayden Turner on 3/7/24.
//

import SwiftUI

class TrainingData: Codable, ObservableObject {
    
    struct Room: Codable, Identifiable {
        struct Point: Codable {
            var id: String
            var rssi: Double
            var time: Double
        }
        
        var id = UUID()
        var room: String
        var points: [Point] = []
        
        mutating func addPoint(point: Point) {
            self.points.append(point)
        }
    }
    
    public var rooms: [Room]
    
    static let header = [
        "room",
        "id",
        "rssi",
        "time"
    ]
    
    init(rooms: [Room] = []) {
        self.rooms = rooms
    }
    
    func addPointToRoom(room: String, id:String, rssi:Double, time:Double) {
        let point = Room.Point(id: id, rssi: rssi, time: time)
        if let ind = rooms.firstIndex(where: { $0.room == room }) {
            var room = rooms[ind]
            room.addPoint(point: point)
            rooms[ind] = room
        } else {
            var room = Room(room: room)
            room.addPoint(point: point)
            rooms.append(room)
        }
    }
    
    func clearRooms() {
        self.rooms = []
    }
    
}

#if DEBUG
extension TrainingData {
    static let example = TrainingData(rooms: [
        TrainingData.Room(room: "Living Room", points: [
            Room.Point(id: "1234567", rssi: 30, time: 12345),
            Room.Point(id: "1234567", rssi: 30, time: 12345),
            Room.Point(id: "1234567", rssi: 30, time: 12345)
        ]),
        TrainingData.Room(room: "Bedroom", points: [
            Room.Point(id: "1234567", rssi: 30, time: 12345),
            Room.Point(id: "1234567", rssi: 30, time: 12345)
        ]),
        TrainingData.Room(room: "Office", points: [
            Room.Point(id: "1234567", rssi: 30, time: 12345)
        ])
    ])
}
#endif
