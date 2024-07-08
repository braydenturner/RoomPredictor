//
//  BluetoothTool.swift
//  RoomPredictor
//
//  Created by Brayden Turner on 3/2/24.
//

import SwiftUI
import CoreBluetooth
import os.log

class BluetoothTool: NSObject, ObservableObject {
    
    
    @Published var collecting: Bool = false
    @Published var trainingFinished: Bool = false
    
    @Published var currentRoom: String = ""
    
    @Published var peripherals: [BluetoothDevice] = []
    
    private var cbManager: CBCentralManager!
    
    override init() {
        super.init()
        self.cbManager = CBCentralManager(delegate: self, queue: .main)
    }
    
    func cleanRoomName() {
        self.currentRoom = self.currentRoom.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    func scanForPeripherals() {
        Logger.default.debug("Starting scan")
        self.cbManager.scanForPeripherals(withServices: nil, options: nil)
    }
    
    func stopScanning() {
        Logger.default.debug("Stopping scan")
        self.cbManager.stopScan()
    }
    
    func saveData(to trainingData: TrainingData) {
        for peripheral in peripherals {
            let id = peripheral.id
            let rssi = peripheral.rssi
            let time = peripheral.time
            
            trainingData.addPointToRoom(room: currentRoom, id:id, rssi:rssi, time: time)
        }
    }
    
    func clearData() {
        peripherals = []
    }
    
}

extension BluetoothTool: CBCentralManagerDelegate {
    
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            Logger.default.debug("Bluetooth is on")
        default:
            break
        }
    }
    
    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String : Any], rssi RSSI: NSNumber) {
        Logger.default.debug("Found item \(peripheral) with RSSI \(RSSI)")
        
        let bluetoothDevice = BluetoothDevice(peripheral: peripheral, rssi: RSSI.doubleValue, time: Date().timeIntervalSinceReferenceDate)
        
        peripherals.append(bluetoothDevice)
    }
}

class BluetoothDevice: NSObject, CBPeripheralDelegate {
    
    let peripheral: CBPeripheral
    
    var rssi: Double
    var time: Double
    var id: String {
        self.peripheral.identifier.uuidString
    }
    
    init(peripheral: CBPeripheral, rssi: Double = -100, time: Double) {
        self.peripheral = peripheral
        self.rssi = rssi
        self.time = time
    }

    func updateRSSI() {
        self.peripheral.readRSSI()
    }

    func peripheral(_ peripheral: CBPeripheral, didReadRSSI RSSI: NSNumber, error: Error?) {
        if let error {
            print(error.localizedDescription)
        }
        self.rssi = Double(truncating: RSSI)
    }
}
