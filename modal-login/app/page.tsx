"use client";
import {
  useAuthModal,
  useLogout,
  useSigner,
  useSignerStatus,
  useUser,
} from "@account-kit/react";
import { useEffect, useState } from "react";

export default function Home() {
  const user = useUser();
  const { openAuthModal } = useAuthModal();
  const signerStatus = useSignerStatus();
  const { logout } = useLogout();
  const signer = useSigner();

  const [createdApiKey, setCreatedApiKey] = useState(false);

  useEffect(() => {
    // User logged out, so reset the state.
    if (!user && createdApiKey) {
      setCreatedApiKey(false);
    }
    // Waiting for user to be logged in.
    if (!user || !signer || !signerStatus.isConnected || createdApiKey) {
      return;
    }

    const submitStamp = async () => {
      const whoamiStamp = await signer.inner.stampWhoami();
      const resp = await fetch("/api/get-api-key", {
        method: "POST",
        body: JSON.stringify({ whoamiStamp }),
      });
      return (await resp.json()) as { publicKey: string };
    };

    const createApiKey = async (publicKey: string) => {
      await signer.inner.experimental_createApiKey({
        // Name of the API key (important b/c the server is checks the prefix).
        name: `server-signer-${new Date().getTime()}`,
        // Public key generated by the server.
        publicKey,
        // Configure how long the API key is valid for.
        expirationSec: 60 * 60 * 24 * 62, // 62 days
      });
    };

    const handleAll = async () => {
      // Submit whoami stamp & get new public key from server.
      const { publicKey } = await submitStamp();
      // Delegate user access to the API key.
      await createApiKey(publicKey);
      // Mark API key as activated in db to flag to consumers that it's ready.
      await fetch("/api/set-api-key-activated", {
        method: "POST",
        body: JSON.stringify({ orgId: user.orgId, apiKey: publicKey }),
      });
      setCreatedApiKey(true);
    };

    handleAll().catch((err) => {
      console.error(err);
      alert("Something went wrong. Please check the console for details.");
    });
  }, [createdApiKey, signer, signerStatus.isConnected, user]);

  // Show alert if crypto.subtle isn't available.
  useEffect(() => {
    if (typeof window === undefined) {
      return;
    }
    try {
      if (typeof window.crypto.subtle !== "object") {
        throw new Error("window.crypto.subtle is not available");
      }
    } catch (err) {
      alert(
        "Crypto api is not available in browser. Please be sure that the app is being accessed via localhost or a secure connection.",
      );
    }
  }, []);

  return (
    <main className="flex min-h-screen flex-col items-center gap-4 justify-center text-center">
      {signerStatus.isInitializing || (user && !createdApiKey) ? (
        <>Loading...</>
      ) : user ? (
        <div className="card">
          <div className="flex flex-col gap-2 p-2">
            <p className="text-xl font-bold">
              YOU ARE SUCCESSFULLY LOGGED IN TO THE GENSYN TESTNET
            </p>
            <button className="btn btn-primary mt-6" onClick={() => logout()}>
              Log out
            </button>
          </div>
        </div>
      ) : (
        <div className="card">
          <p className="text-xl font-bold">LOGIN TO THE GENSYN TESTNET</p>
          <div className="flex flex-col gap-2 p-2">
            <button className="btn btn-primary mt-6" onClick={openAuthModal}>
              Login
            </button>
          </div>
        </div>
      )}
    </main>
  );
}