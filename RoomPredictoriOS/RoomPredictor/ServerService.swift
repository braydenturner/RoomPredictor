//
//  ServerService.swift
//  RoomPredictor
//
//  Created by Brayden Turner on 3/7/24.
//

import Foundation


struct ServerService {
    
    static let url: URL = URL(string: "http://")!
    
    
    func send(data: TrainingData) async {
        
        guard let encoded = try? JSONEncoder().encode(data) else {
            print("Failed to encode order")
            return
        }
        
        let parameters: [String: String] = ["name": "", "password": ""]
        
        //now create the URLRequest object using the url object
        var request = URLRequest(url: Self.url)
        request.httpMethod = "POST" //set http method as POST
        
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue("application/json", forHTTPHeaderField: "Accept")
        
        do {
            let (data, _) = try await URLSession.shared.upload(for: request, from: encoded)
            // handle the result
        } catch {
            print("Sending data failed: \(error.localizedDescription)")
        }
    }

}
