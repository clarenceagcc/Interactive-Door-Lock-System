import RPi.GPIO as GPIO
import time

RELAY_PIN = 21
class DoorLock:
	def __init__(self):
		self.setup()
		
	def setup(self):
		GPIO.setmode(GPIO.BCM)
		GPIO.setup(RELAY_PIN, GPIO.OUT)
		GPIO.output(RELAY_PIN, GPIO.OUT)
		self.lock() # Start in locked state
		print("Door Lock System Initialize")
		
	def lock(self):
		print("Locking door...")
		GPIO.output(RELAY_PIN, GPIO.LOW)
		print("Door Locked")
		
	def unlock(self):
		print("Unlocking door...")
		GPIO.output(RELAY_PIN, GPIO.HIGH)
		print("Door Unlocked")
		
	def test_cycle(self):
		print("Running a test cycle...")
		self.lock()
		time.sleep(2)
		
		self.unlock()
		time.sleep(5)
		
		self.lock()
		print("Test cycle complete")
		
	def cleanup(self):
		self.lock()
		GPIO.cleanup()
		print("GPIO cleanup complete")

if __name__ == "__main__":
	lock_system = DoorLock()
	try:
		while True:
			command = input("\nEnter Command (unlock/lock/test/exit): ").lower()
			if command == "unlock":
				lock_system.unlock()
			elif command == "lock":
				lock_system.lock()
			elif command == "test":
				lock_system.test_cycle()
			elif command == "exit":
				break
			else:
				print("Invalid")
				
	except KeyboardInterrupt:
		print("\nProgram stopped")
	finally:
		cleanup()
