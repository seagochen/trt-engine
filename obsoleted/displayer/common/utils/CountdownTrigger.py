class CountdownTrigger:

    def __init__(self, trigger_count=10):
        self.countdown_trigger = 0
        self.trigger_count = trigger_count

    def perform(self):

        if self.countdown_trigger == 0:
            # リセットtrigger
            self.countdown_trigger = self.trigger_count
            return True

        # カウントダウン
        self.countdown_trigger -= 1
        return False



if __name__ == "__main__":

    trigger = CountdownTrigger(3)

    for i in range(10):
        if trigger.perform():
            print("Triggered")
        else:
            print("Not Triggered")

