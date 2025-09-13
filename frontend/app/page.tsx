"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Eye, Hand, Camera, Mic, Volume2 } from "lucide-react"
import { useState } from "react"

export default function HomePage() {
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  const [isStarting, setIsStarting] = useState(false)
  const [isStopping, setIsStopping] = useState(false)

  const startCamera = async () => {
    try {
      setIsStarting(true)
      await fetch(`${apiBaseUrl}/start`)
    } catch (e) {
      console.error(e)
    } finally {
      setIsStarting(false)
    }
  }

  const stopCamera = async () => {
    try {
      setIsStopping(true)
      await fetch(`${apiBaseUrl}/stop`)
    } catch (e) {
      console.error(e)
    } finally {
      setIsStopping(false)
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-background/95 backdrop-blur sticky top-0 z-50">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-center">
            <div className="flex items-center space-x-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                <Eye className="h-6 w-6" />
              </div>
              <h1 className="text-2xl font-bold text-foreground">VisionBridge</h1>
            </div>
          </div>
        </div>
      </header>

      <main className="py-8 bg-background" role="main">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-4xl">
          <div className="text-center space-y-6 mb-12">
            <h2 className="text-3xl lg:text-4xl font-bold text-foreground text-balance">
              AI Assistant for Independence
            </h2>
            <p className="text-xl text-muted-foreground text-pretty leading-relaxed max-w-2xl mx-auto">
              Use your camera to identify objects around you or translate sign language into speech. Simple, fast, and
              designed for accessibility.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6 mb-12">
            {/* Object Detection Card */}
            <Card className="border-2 border-primary/20 bg-card hover:border-primary/40 transition-all focus-within:border-primary">
              <CardHeader className="text-center space-y-4 pb-6">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 text-primary mx-auto">
                  <Camera className="h-8 w-8" />
                </div>
                <CardTitle className="text-2xl text-card-foreground">Identify Objects</CardTitle>
                <CardDescription className="text-lg text-muted-foreground">
                  Point your camera at anything and hear what it is
                </CardDescription>
              </CardHeader>
              <CardContent className="text-center space-y-4">
                <Button
                  size="lg"
                  className="w-full bg-primary hover:bg-primary/90 text-primary-foreground text-lg py-6"
                  onClick={startCamera}
                  disabled={isStarting}
                  aria-label="Start object detection - will use your camera to identify objects and speak their names"
                >
                  <Camera className="mr-3 h-5 w-5" />
                  {isStarting ? "Starting..." : "Start Camera"}
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  className="mt-2"
                  onClick={stopCamera}
                  disabled={isStopping}
                  aria-label="Stop object detection"
                >
                  {isStopping ? "Stopping..." : "Stop Camera"}
                </Button>
                <div className="flex items-center justify-center space-x-2 text-sm text-muted-foreground">
                  <Volume2 className="h-4 w-4" />
                  <span>Audio descriptions included</span>
                </div>
              </CardContent>
            </Card>

            {/* Sign Language Translation Card */}
            <Card className="border-2 border-accent/20 bg-card hover:border-accent/40 transition-all focus-within:border-accent">
              <CardHeader className="text-center space-y-4 pb-6">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-accent/10 text-accent mx-auto">
                  <Hand className="h-8 w-8" />
                </div>
                <CardTitle className="text-2xl text-card-foreground">Translate Sign Language</CardTitle>
                <CardDescription className="text-lg text-muted-foreground">
                  Convert sign language gestures to spoken words
                </CardDescription>
              </CardHeader>
              <CardContent className="text-center space-y-4">
                <Button
                  size="lg"
                  className="w-full bg-accent hover:bg-accent/90 text-accent-foreground text-lg py-6"
                  aria-label="Start sign language translation - will watch for sign language and speak the translation"
                >
                  <Hand className="mr-3 h-5 w-5" />
                  Start Translation
                </Button>
                <div className="flex items-center justify-center space-x-2 text-sm text-muted-foreground">
                  <Mic className="h-4 w-4" />
                  <span>Real-time voice output</span>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="bg-muted/30 rounded-lg p-6 text-center space-y-4">
            <h3 className="text-xl font-semibold text-foreground">How to Use</h3>
            <div className="grid md:grid-cols-2 gap-6 text-left">
              <div className="space-y-2">
                <h4 className="font-medium text-foreground flex items-center">
                  <Camera className="h-4 w-4 mr-2" />
                  Object Detection
                </h4>
                <p className="text-muted-foreground">
                  Tap "Start Camera", point your phone at any object, and listen to the audio description.
                </p>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium text-foreground flex items-center">
                  <Hand className="h-4 w-4 mr-2" />
                  Sign Language
                </h4>
                <p className="text-muted-foreground">
                  Tap "Start Translation", show sign language to the camera, and hear the spoken translation.
                </p>
              </div>
            </div>
          </div>

          <div className="text-center mt-12 pt-8 border-t border-border">
            <p className="text-muted-foreground">
              Need help? Contact us at{" "}
              <a href="mailto:support@visionbridge.app" className="text-primary hover:underline">
                support@visionbridge.app
              </a>
            </p>
          </div>
        </div>
      </main>
    </div>
  )
}
