import serial
import time


def read_weight():
    ser = serial.Serial('COM10', 9600, timeout=1)  # Change COM3 to your Arduino port
    ser.flush()

    while True:
        if ser.in_waiting > 0:
            weight_data = ser.readline().decode('utf-8').strip()
            try:
                weight = float(weight_data.split()[1])  # Extract the numerical value
                print(f"Weight: {weight} g")
            except ValueError:
                print(f"Invalid data received: {weight_data}")
        time.sleep(1)


if __name__ == "__main__":
    print("Reading weight from HX711 sensor...")
    read_weight()
