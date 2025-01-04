//
//  RoomPredictorApp.swift
//  RoomPredictor
//
//  Created by Brayden Turner on 2/9/24.
//

import SwiftUI

@main
struct RoomPredictorApp: App {
    
    @State var bluetoothTool: BluetoothTool = BluetoothTool()
    
    var body: some Scene {
        WindowGroup {
            ContentView(bluetoothTool: bluetoothTool)
        }
    }
}
