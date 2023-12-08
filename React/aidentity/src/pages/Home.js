import React from "react";
import UserHistory from "../components/UserHistory";
import UserPredict from "../components/UserPredict";

function Home() {
  return (
    <div>
      <UserPredict />
      <UserHistory />
    </div>
  );
}

export default Home;
