//
//  CurrentRoom.swift
//  RoomPredictor
//
//  Created by Brayden Turner on 3/2/24.
//

import SwiftUI

struct CurrentRoom: View {
    
    @Bindable var bluetoothTool: BluetoothTool
    
    var body: some View {
            TextField("Room Name", text: $bluetoothTool.currentRoom)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .hidden(bluetoothTool.collecting)
            Text("\(bluetoothTool.peripherals.count) data points collected for \(bluetoothTool.currentRoom)")
                .font(.headline)
                .hidden(!bluetoothTool.collecting)
    }
    
}

extension View {
    func hidden(_ shouldHide: Bool) -> some View {
        opacity(shouldHide ? 0 : 1)
    }
}
