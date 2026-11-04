
import socket
import csv
import time

from pathlib import Path


HOST = "127.0.0.1"
PORT = 5500
INTERVAL = 1.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "flightgear_recording.csv"

PROPERTIES = {
    "altitude_ft": "/position/altitude-ft",
    "airspeed_kt": "/velocities/airspeed-kt",
    "heading_deg": "/orientation/heading-deg",
}


def get_property(connection, reader, property_path):
    command = f"get {property_path}\r\n"
    connection.sendall(command.encode("utf-8"))

    response = reader.readline().strip()

    return float(response)


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("AEROMENTOR - FLIGHT RECORDER")
    print("----------------------------")
    print(f"Output: {OUTPUT_FILE}")
    print("Recording every second.")
    print("Press CTRL+C to stop.\n")

    try:
        with socket.create_connection(
            (HOST, PORT), timeout=5
        ) as connection:

            connection.settimeout(5)

            reader = connection.makefile(
                "r", encoding="utf-8", newline="\n"
            )

            connection.sendall(b"data\r\n")

            start_time = time.monotonic()
            sample_count = 0

            with open(
                OUTPUT_FILE,
                "w",
                newline="",
                encoding="utf-8",
            ) as csv_file:

                writer = csv.writer(csv_file)

                writer.writerow([
                    "time_s",
                    "altitude_ft",
                    "airspeed_kt",
                    "heading_deg",
                ])

                while True:
                    sample_start = time.monotonic()

                    elapsed = sample_start - start_time

                    altitude = get_property(
                        connection,
                        reader,
                        PROPERTIES["altitude_ft"],
                    )

                    airspeed = get_property(
                        connection,
                        reader,
                        PROPERTIES["airspeed_kt"],
                    )

                    heading = get_property(
                        connection,
                        reader,
                        PROPERTIES["heading_deg"],
                    )

                    writer.writerow([
                        round(elapsed, 2),
                        altitude,
                        airspeed,
                        heading,
                    ])

                    csv_file.flush()

                    sample_count += 1

                    print(
                        f"Time: {elapsed:7.2f}s | "
                        f"Altitude: {altitude:8.2f} ft | "
                        f"Airspeed: {airspeed:6.2f} kt | "
                        f"Heading: {heading:6.2f}°"
                    )

                    remaining = INTERVAL - (
                        time.monotonic() - sample_start
                    )

                    if remaining > 0:
                        time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nRecording stopped.")

    except (OSError, ValueError) as error:
        print(f"\nTelemetry error: {error}")

    finally:
        print(f"CSV location: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
